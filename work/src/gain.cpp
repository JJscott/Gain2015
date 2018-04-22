
// std
#include <iostream>
#include <vector>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// project
#include "patchmatch.hpp"
#include "terrain.hpp"


using namespace cv;
using namespace std;


namespace gain {

	void pyrUpJitter(Mat nnf, Mat heightoffset, Mat nnfdest, Mat heightoffsetdest, float jitter) {
		assert(nnf.type() == CV_32SC2);
		assert(nnfdest.type() == CV_32SC2);
		assert(heightoffset.type() == CV_32FC1);
		assert(heightoffsetdest.type() == CV_32FC1);
		assert(nnf.size() == heightoffset.size());
		assert(nnfdest.size() == heightoffsetdest.size());
		assert(nnf.size() == nnfdest.size() / 2);

		for (int i = 0; i < nnfdest.rows; i++) {
			for (int j = 0; j < nnfdest.cols; j++) {
				Point p(j / 2, i / 2);
				Point o(j % 2, i % 2);
				nnfdest.at<Point>(i, j) = nnf.at<Point>(p) * 2 + o;
				heightoffsetdest.at<Point>(i, j) = heightoffset.at<Point>(p);
			}
		}
	}



	Mat appearance_space(Mat data) {
		assert(data.type() == CV_32FC1);

		const int numComponents = 4;

		//Mat gaussian1d = getGaussianKernel(5, 1);
		//Mat gaussian2d = gaussian1d * gaussian1d.t();

		// hardcoded normalized 5x5 gaussian kernal
		const float guassian_kernal[5][5] = {
			{ 0.00296902, 0.0133062, 0.0219382,  0.0133062, 0.00296902 },
			{ 0.0133062,  0.0596343, 0.0983203,  0.0596343, 0.0133062  },
			{ 0.0219382,  0.0983203, 0.16210312, 0.0983203, 0.0219382  },
			{ 0.0133062,  0.0596343, 0.0983203,  0.0596343, 0.0133062  },
			{ 0.00296902, 0.0133062, 0.0219382,  0.0133062, 0.00296902 }
		};

		// create the PCA set
		Mat neighbourhood_data(data.size(), CV_32FC(25));
		for (int i = 0; i < data.rows; i++) {
			for (int j = 0; j < data.cols; j++) {
				Point p(j, i);
				for (int ii = 0; ii < 5; ii++) {
					for (int jj = 0; jj < 5; jj++) {
						Point q(max(0, min(j + jj - 2, data.cols - 1)), max(0, min(p.y + ii - 2, data.rows - 1)));
						neighbourhood_data.at<Vec<Vec<float, 5>, 5>>(p)[ii][jj] = data.at<float>(q) * guassian_kernal[ii][jj];
					}
				}
			}
		}
		Mat pcaset = neighbourhood_data.reshape(1, data.cols * data.rows);

		// compute mean and PCA
		Mat pcaset_mean;
		PCA pca(pcaset, Mat(), PCA::DATA_AS_ROW, numComponents);
		reduce(pcaset, pcaset_mean, 1, CV_REDUCE_SUM);

		// compute mean of eigen vectors TODO
		//Mat eigen_sum(numComponents, 1, CV_32F);
		//reduce(pcaset, pcaset_mean, 1, CV_REDUCE_SUM);
		//for (int r = 0; r < 4; r++) *eigen_mean.ptr<float>(r) = norm(pca.eigenvectors.row(r));
		//eigen_mean = eigen_mean.reshape(numComponents, 1);


		// transform to reduced eigen space
		Mat reduced = pcaset * pca.eigenvectors.t();

		// manually set the mean (TODO investigate scaling of 4 components instead of setting of just 1)
		for (int i = 0; i < reduced.rows; i++) {
			(*reduced.ptr<Vec<float, 4>>(i))[0] = *pcaset_mean.ptr<float>(i);
		}

		// reshape and return
		Mat final = reduced.reshape(numComponents, data.rows);
		return final;
	}




	void correction(Mat example, Mat app_space, Mat synth, Mat heightoffset) {
		assert(synth.type() == CV_32SC2);
		assert(heightoffset.type() == CV_32FC1);
		assert(example.size() == app_space.size());
		assert(synth.size() == heightoffset.size());



		//using app_vec_t = appearance.value_t;
		using app_vec_t = basic_vec<float, A>;


		// delta set for diagonal neighbours
		const Point diagonal_delta[]{{-1, -1}, { 1, -1}, {-1, 1}, {1, 1}};


		// matrix set (for diagonal neighbour prediction)
		const imat2 diagonal_mat[]{
			imat2{ ivec2(0,0),ivec2(0,0) },
			imat2{ ivec2(1,0),ivec2(0,0) },
			imat2{ ivec2(0,0),ivec2(0,1) }
		};


		//  delta set for 3x3 neighbourhood
		const ivec2 neighbour_delta[]{
			ivec2{ -1, -1 },ivec2{ -1, 0 },ivec2{ -1, 1 },
			ivec2{ 0, -1 }, ivec2{ 0, 0 }, ivec2{ 0, 1 },
			ivec2{ 1, -1 }, ivec2{ 1, 0 }, ivec2{ 1, 1 }
		};



		// debug
		Mat cost_arr;
		cost_arr.create(synth.size(), CV_32F);

		// 4 subpasses
		const Point subpass_delta[]{ Point{ 0,0 }, Point{ 1,0 },Point{ 0,1 },Point{ 1,1 } };

		for (int subpass = 0; subpass < 4; ++subpass) {

			//#pragma omp parallel for
			for (int j = 0; j < synth.cols - 1; j += 2) {
				for (int i = 0; i < synth.rows - 1; i += 2) {

					// the pixel we want to replace and it's neighbourhood
					Point p = Point(i, j) + subpass_delta[subpass];
					Mat np(1, 4, app_space.type());

					// mask for height offset to be combined with appearance vectors
					app_vec_t a{ 0 };
					a[0] = 1;

					// for every diagonal neighbour
					// compute an approximate appearance space vector and 
					for (int n = 0; n < 4; ++n) {
						// neighbour pixel
						Point u = p + diagonal_delta[n];

						// take neighbour and 2 adjacent locations in appearance space
						// average them out as if we were predicting the change to this neighbour
						// for future subpasses that rely on it
						for (const Mat &mat : diagonal_mat) {
							// we use sample_bilinear because we are dealing with coordinates that
							// are not integer coords. meaning each sample will actually be bilinearly
							// interpolated between each neighbour
							ivec2 v = mat * diagonal_delta[n]; // adjacent location offset (0,0), (1,0), (0,1)
							vec2 s = synthesis.at_clamp(u + v); // the coord at the neighbours adjacent location
							np[n] += appearance.sample_bilinear(s - v)  // an approximate appearance vector (for future value of u+v)
								+ (height_offset.at_clamp(u + v) * a); // combined with the associated height offset (for current value of u+v)
						}
						np[n] /= 3; // average out

						// otherwise we would do something like this instead
						//np[n] = appearance.sample_bilinear(synthesis.at_clamp(u)) + (height_offset.at_clamp(u) * a);
					}



					//
					// Find the best matching Neighbourhood
					//

					// default
					Vec2f best_q;
					float best_h;
					float best_cost = numeric_limits<float>::infinity();

					// for self and all neighbours
					// use k-cohearance to find a good match
					for (int i = 0; i < 9; ++i) {
						// convert half int coord to int coord
						ivec2 s = floor(synthesis.at_clamp(p + neighbour_delta[i]));

						// the k-cohearence candidates for coord s minus delta
						basic_vec<vec2, K> &candidates = cohearence.at_clamp(ivec2(s - neighbour_delta[i]));
						for (size_t k = 0; k < K; ++k) {


							// the candidate 'q' with some height offset h, and its neighbourhood
							ivec2 q = candidates[k];
							float h = 0;
							basic_vec<app_vec_t, 4> nq;

							// for diagonal neighbours
							for (int n = 0; n < 4; ++n) {
								nq[n] = appearance.at_clamp(q + diagonal_delta[n]); // record appearance vector for neighbour
								h += (np[n][0] - nq[n][0]); // accumulate an height difference between the neighbourhoods
							}

							// average out h and add to q's neighbourhood
							// this minimizes the height difference between candidate and target
							h /= 4;
							//h = 0; // uncomment for no height offset in the correction
							for (int n = 0; n < 4; ++n)
								nq[n][0] += h;



							// compute the cost
							// ssd of nq and np
							float cost = 0;
							for (int n = 0; n < 4; ++n) {
								app_vec_t d = np[n] - nq[n];
								cost += dot(d, d);
							}

							// if not the the lowest k-value, double the cost
							//if (k > 0) cost *= 1.5;
							if (k > 0) cost *= 2;

							// find best correction
							if (cost < best_cost) {
								best_cost = cost;
								best_h = h;
								best_q = vec2(q) + 0.5; // half int coordinates
							}
						}
					}

					synthesis.at(p) = best_q;
					height_offset.at(p) = best_h;
					cost_arr.at(p) = best_cost;

				}
			}


		} // subpasses

			// debug
		auto height_img = arr_to_heightmap(resolve_coord(synthesis, example, height_offset));
		auto error_img = arr_to_heatmap(cost_arr);

		//for (int x = 0; x < error_img.width(); x++) {
		//	for (int y = 0; y < error_img.height(); y++) {
		//		error_img.texel(x, y) = vec3(((height_img.texel(x, y) + 1) / 2).x) / 2 + error_img.texel(x, y) / 2;
		//	}
		//}

		util::cache_upload(error_img, stringf("cost_arr pass=", pass, " | min=", cost_arr.min(), " : max=", cost_arr.max()));
		




	
	}







	Mat synthesizeTerrain(Mat example, Size synth_size, int example_levels, int synth_levels) {

		assert(synth_levels > example_levels); // M > L

		// create exemplar pyramid
		vector<Mat> example_pyramid(example_levels);
		buildPyramid(example, example_pyramid, example_levels);

		// create synthesis pyramid
		Mat synth(synth_size.height, synth_size.width, CV_32FC2);
		Mat heightoffset(synth_size.height, synth_size.width, CV_32FC1, Scalar(0));
		vector<Mat> synth_pyramid(synth_levels);
		vector<Mat> heightoffset_pyramid(synth_levels);
		buildPyramid(synth, synth_pyramid, example_levels);
		buildPyramid(heightoffset, heightoffset_pyramid, example_levels);


		// initialization
		synth_pyramid[synth_levels - 1].setTo(Scalar(0, 0));


		// iteration
		for (int level = synth_levels - 1; level-- > 0;) {

			// upsample / jitter
			pyrUpJitter(
				synth_pyramid[level - 1],
				heightoffset_pyramid[level - 1],
				synth_pyramid[level],
				heightoffset_pyramid[level],
				0.4
			);

			// correction
			if (level < example_levels) {
				Mat app_space = appearance_space(example_pyramid[level]);
				correction(example_pyramid[level], app_space, synth_pyramid[level], heightoffset_pyramid[level]);
			}
		}

		return example; // TODO change

	}



	void test(Mat testimage) {




		//// Appearance-space
		//Mat data(testimage.size(), CV_32FC1);
		//testimage.convertTo(data, CV_32FC1);

		//Mat reduced = appearance_space(data);

		//Mat ind[4];
		//split(reduced, ind);

		//imwrite("output/appearance_0.exr", ind[0]);
		//imwrite("output/appearance_1.exr", ind[1]);
		//imwrite("output/appearance_2.exr", ind[2]);
		//imwrite("output/appearance_3.exr", ind[3]);




		//Mat terrain = reduced.clone();



		//// K=2 Patchmatch
		//Mat nnf = k2_patchmatch(terrain, terrain, 5, 4);

		//// convert offset coordinates to color image
		//Mat nnf_img_k1(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
		//Mat nnf_img_k2(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
		//for (int r = 0; r < nnf.rows; r++) {
		//	auto in_row = nnf.ptr<Vec<Vec2f, 2>>(r);
		//	auto out_row_k1 = nnf_img_k1.ptr<Vec3b>(r);
		//	auto out_row_k2 = nnf_img_k2.ptr<Vec3b>(r);
		//	for (int c = 0; c < nnf.cols; c++) {
		//		out_row_k1[c][2] = int((255.0 * in_row[c][0][0]) / terrain.cols); // cols -> r
		//		out_row_k1[c][1] = int((255.0 * in_row[c][0][1]) / terrain.rows); // rows -> g

		//		out_row_k2[c][2] = int((255.0 * in_row[c][1][0]) / terrain.cols); // cols -> r
		//		out_row_k2[c][1] = int((255.0 * in_row[c][1][1]) / terrain.rows); // rows -> g
		//	}
		//}

		//imwrite("output/NNF_k1.png", nnf_img_k1);
		//imwrite("output/NNF_k2.png", nnf_img_k2);




		//terrain test = terrainReadTIFF("work/res/southern_alps_s045e169.tif");
		//terrain test = terrainReadTIFF("work/res/mt_fuji_n035e138.tif"); 
		//imwrite("output/test_img1.png", heightmapToImage(test.heightmap));

	}

}