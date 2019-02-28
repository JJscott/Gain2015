// std
#include <iostream>
#include <vector>
#include <queue>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// project
#include "terrainopt.hpp"
#include "terrain.hpp"
#include "patchmatch.hpp"
#include "opencv_util.hpp"


using namespace cv;
using namespace std;


namespace{
	// Pixel & pixel, const int * position
	Mat upsampleNNF(Mat nnf, Size size) {
		Mat up(size, CV_32FC2);

		for (int i = 0; i < up.rows; i++) {
			for (int j = 0; j < up.cols; j++) {
				Vec2f o(j % 2, i % 2);
				up.at<Vec2f>(i, j) = (nnf.at<Vec2f>(i / 2.0, j / 2.0) * 2.0) + o;
			}
		}
		return up;
	}
}


namespace scott2 {


	Mat init_patches(
		Mat example_heightmap,
		Mat target_heightmap,
		int patch_size
	) {
		assert(example_heightmap.type() == target_heightmap.type());

		// create candidates
		vector<Mat> candidates;
		for (int i = 0; i < example_heightmap.rows - patch_size; i += patch_size / 2) {
			for (int j = 0; j < example_heightmap.cols - patch_size; j += patch_size / 2) {
				candidates.push_back(example_heightmap(Range(i, i + patch_size), Range(j, j + patch_size)));
			}
		}

		// create output
		Mat synthesis(target_heightmap.size(), target_heightmap.type());
		copyMakeBorder(target_heightmap, synthesis, 0, patch_size, 0, patch_size, BORDER_REPLICATE);

		// select patches for output
		for (int i = 0; i < synthesis.rows - patch_size; i += patch_size) {
			for (int j = 0; j < synthesis.cols - patch_size; j += patch_size) {
				Mat target = synthesis(Range(i, i + patch_size), Range(j, j + patch_size));

				// find best replacement
				Mat best_cand;
				double best_cost = numeric_limits<float>::infinity();
				for (Mat &cand : candidates) {
					double cost = norm(target, cand);
					if (cost < best_cost) {
						best_cost = cost;
						best_cand = cand;
					}
				}
				best_cand.copyTo(target);
			}
		}
		
		return Mat(synthesis(Range(0, target_heightmap.rows), Range(0, target_heightmap.cols)));
	}


	void test(Mat testimage) {


		const int patch_size = 7;
		const int patchmatch_iter = 4;
		const int pyramid_depth = 4;
		const int correction_iter = 4;


		// create the guess for the target
		//Mat source_img = testimage.clone();
		Mat source_img = gain::terrainReadTIFF("work/res/mt_fuji_n035e138.tif").heightmap;
		//Mat source_img = gain::terrainReadTIFF("work/res/southern_alps_s045e169.tif").heightmap;
		double smin, smax;
		minMaxIdx(source_img, &smin, &smax);
		Mat target_img = gain::terrainReadImage("work/res/multi_fractal.png", smin, smax, 20).heightmap;
		//Mat target_img = gain::terrainReadImage("work/res/half_life_blur.png", smin, smax, 20).heightmap;
		//Mat target_img = gain::terrainReadImage("work/res/Result.png", smin, smax, 20).heightmap;
		//Mat target_img = gain::terrainReadImage("work/res/half_life.png", smin, smax, 20).heightmap;


		//Mat synth_img = init_patches(source_img, target_img, 30);
		Mat synth_img = target_img;

		//Mat nnf = patchmatch(source_img, synth_img, patch_size, patchmatch_iter);
		//imwrite("output/NNF.png", nnfToImg(nnf, source_img.size()));
		//imwrite("output/Result.png", gain::heightmapToImage(synth_img));

		//Mat synth_img = source_img.clone();


		// construct pyramids
		vector<Mat> source_pyr, synth_pyr;
		buildPyramid(source_img, source_pyr, pyramid_depth);
		buildPyramid(synth_img, synth_pyr, pyramid_depth);


		////initialize lowest pyramid
		//Mat nnf(synth_pyr[pyramid_depth - 1].rows, synth_pyr[pyramid_depth - 1].cols, CV_32FC2);
		//for (int i = 0; i < nnf.rows; i++) {
		//	for (int j = 0; j < nnf.cols; j++) {
		//		nnf.at<Vec2f>(i, j) = Vec2f(
		//			util::random<float>(patch_size / 2, source_pyr[pyramid_depth - 1].cols - patch_size / 2),
		//			util::random<float>(patch_size / 2, source_pyr[pyramid_depth - 1].rows - patch_size / 2)
		//		);
		//	}
		//}
		//synth_pyr[pyramid_depth - 1] = reconstruct(source_pyr[pyramid_depth - 1], nnf, patch_size);

		imwrite(util::stringf("output/img_init.png"), gain::heightmapToImage(synth_pyr[pyramid_depth - 1]));
		//imwrite(util::stringf("output/nnf_init.png"), nnfToImg(nnf, synth_pyr[pyramid_depth - 1].size()));

		// iteration over pyramid
		cout << "Begin synthesis" << endl;
		Mat nnf;
		for (int k = pyramid_depth; k-->0;) {
			cout << " - Pyramid Iterations " << (pyramid_depth - k) << "/" << pyramid_depth << endl;
			// correction
			for (int i = 0; i < correction_iter; i++) {
				cout << "   - Correction Iterations " << (i + 1) << "/" << correction_iter << endl;
				nnf = patchmatch(source_pyr[k], synth_pyr[k], patch_size, patchmatch_iter, nnf);
				synth_pyr[k] = reconstruct(source_pyr[k], nnf, patch_size);

				imwrite(util::stringf("output/img_", (pyramid_depth - k), "_", i, ".png"), gain::heightmapToImage(synth_pyr[k]));
				imwrite(util::stringf("output/nnf_", (pyramid_depth - k), "_", i, ".png"), nnfToImg(nnf, source_pyr[k].size()));
			}

			// upsample
			if (k > 0) {
				nnf = patchmatch(source_pyr[k], synth_pyr[k], patch_size, patchmatch_iter);
				nnf = upsampleNNF(nnf, synth_pyr[k - 1].size());
				synth_pyr[k - 1] = reconstruct(source_pyr[k - 1], nnf, patch_size);
			}
		}
		cout << "Finished" << endl;

		// reconstruct result
		Mat result = gain::heightmapToImage(reconstruct(source_img, nnf, patch_size));
		Mat nnf_img = nnfToImg(nnf, source_img.size());

		imwrite("output/NNF.png", nnf_img);
		imwrite("output/Result.png", result);
	}
}