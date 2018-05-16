
// std
#include <iostream>
#include <vector>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// project
#include "patchmatch.hpp"
#include "opencv_util.hpp"
#include "terrain.hpp"


using namespace cv;
using namespace std;


namespace gain {

	void pyrUpJitter(Mat nnf, Mat heightoffset, Mat nnfdest, Mat heightoffsetdest, float jitter_mag) {
		assert(nnf.type() == CV_32FC2);
		assert(nnfdest.type() == CV_32FC2);
		assert(heightoffset.type() == CV_32FC1);
		assert(heightoffsetdest.type() == CV_32FC1);
		assert(nnf.size() == heightoffset.size());
		assert(nnfdest.size() == heightoffsetdest.size());
		assert(nnf.size() == nnfdest.size() / 2);

		for (int i = 0; i < nnfdest.rows; i++) {
			for (int j = 0; j < nnfdest.cols; j++) {
				Point p(j / 2, i / 2);
				Vec2f o(j % 2, i % 2);
				Vec2f jitter(util::random<float>(-1, 1), util::random<float>(-1, 1));
				nnfdest.at<Vec2f>(Point(j, i)) = nnf.at<Vec2f>(p) * 2 + o + (jitter * jitter_mag);
				heightoffsetdest.at<float>(Point(j, i)) = heightoffset.at<float>(p);
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
						Point q(clamp(j + jj - 2, 0, data.cols - 1), clamp(i + ii - 2, 0, data.rows - 1));
						// TODO check this shit
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




	void correction(Mat example, Mat app_space, Mat coherence, Mat synth, Mat heightoffset) {
		assert(synth.type() == CV_32FC2);
		assert(heightoffset.type() == CV_32FC1);
		assert(example.size() == app_space.size());
		assert(synth.size() == heightoffset.size());

		// specific
		assert(app_space.type() == CV_32FC4);
		assert(coherence.type() == CV_32FC4);

		// delta set for diagonal neighbours
		const Point diagonal_delta[]{{-1, -1}, { 1, -1}, {-1, 1}, {1, 1}};

		//  delta set for 3x3 neighbourhood
		const Point neighbour_delta[]{
			{ -1, -1 },{ -1, 0 },{ -1, 1 },
			{ 0, -1 }, { 0, 0 }, { 0, 1 },
			{ 1, -1 }, { 1, 0 }, { 1, 1 }
		};

		// blank matrix for constructing appearnace space thing for height offset
		const Mat np_temp_zero = Mat::zeros(3, 4, CV_32FC1);



		// debug
		Mat cost_arr;
		cost_arr.create(synth.size(), CV_32F);

		// 4 subpasses
		const Point subpass_delta[]{{ 0,0 }, { 1,0 },{ 0,1 }, { 1,1 }};

		for (int subpass = 0; subpass < 4; ++subpass) {

			//#pragma omp parallel for
			for (int j = 0; j < synth.cols; j += 2) {
				for (int i = 0; i < synth.rows; i += 2) {

					//
					// Get the Pixel/Neighbourhood we are correcting
					//

					// the pixel we are correcting
					Point p = Point(j, i) + subpass_delta[subpass];
					if (p.y >= synth.rows || p.x >= synth.cols) continue; // bound check

					// construct the neighbourhood
					Mat np(1, 4, app_space.type(), Scalar(0, 0, 0, 0));
					// temporary neighbourhood for sampling/averaging values
					Mat np_temp_neighbour_coords(3, 4, CV_16SC2); // coordinates of neighbours and their adjacent offset neighbours
					Mat np_temp_synth_coords(3, 4, synth.type()); // values in synth retrieved via np_temp_neighbour_coords
					Mat np_temp_app(3, 4, app_space.type()); // values in app_space retrieved via np_temp_synth_coords
					Mat np_temp_ho_single(3, 4, heightoffset.type()); // values in heightoffset retrieved via np_temp_neighbour_coords
					Mat np_temp_ho_multi(3, 4, app_space.type()); // space out values (h, 0, 0, 0) from np_temp_ho_single

					// for every diagonal neighbour
					for (int n = 0; n < 4; ++n) {
						Point u = p + diagonal_delta[n];
						Point vv[]{ // adjacent location offset (0,0), (+-1,0), (0,+-1)
							{ 0, 0 },
							{ diagonal_delta[n].x, 0 },
							{ 0, diagonal_delta[n].y }
						};

						// 3x4 matrix, each column is a diagonal neighbour and its adjacent offset neighbours
						for (int nn = 0; nn < 3; nn++) {
							//TODO some problem here with setting mapped values
							Point temp = u + vv[nn];
							np_temp_neighbour_coords.at<Vec2s>(nn, n) = Vec2s(temp.x, temp.y);
						}
					}

					// fill out the temp matricies
					remap(synth, np_temp_synth_coords, np_temp_neighbour_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);
					remap(app_space, np_temp_app, np_temp_synth_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);
					remap(heightoffset, np_temp_ho_single, np_temp_neighbour_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);
					merge(vector<Mat>{ np_temp_ho_single, np_temp_zero, np_temp_zero, np_temp_zero }, np_temp_ho_multi);

					// average appearance space combine with height offset 
					np_temp_app += np_temp_ho_multi;
					reduce(np_temp_app, np, 0, CV_REDUCE_AVG);


					cout << "p" << p << endl;
					cout << "np" << endl;
					cout << np << endl;


					//
					// Find the best matching Pixel/Neighbourhood
					//

					// collect candidates of neighbours
					Mat candidates_neighbour_coords(1, 9, CV_16SC2); // coordinates of neighbours in synth
					Mat candidates_neighbour_offset(1, 9, synth.type()); // the backward translation of neighbours in synth
					Mat candidates_synth_coords(1, 9, synth.type()); // values in synth retrieved via candidates_neighbour_coords
					Mat candidates_full(1, 9, coherence.type()); // values in candidates retrieved via (candidates_synth_coords - candidates_neighbour_offset)

					// TODO
					for (int i = 0; i < 9; ++i) {
						Point temp = clampToMat(p + neighbour_delta[i], synth);
						candidates_neighbour_coords.at<Vec2s>(0, i) = Vec2s(temp.x, temp.y);
						candidates_neighbour_offset.at<Vec2f>(0, i) = -Vec2f(neighbour_delta[i].x, neighbour_delta[i].y);
					}

					// TODO
					remap(synth, candidates_synth_coords, candidates_neighbour_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);
					candidates_synth_coords -= candidates_neighbour_offset;
					remap(coherence, candidates_full, candidates_synth_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);


					// current best
					Vec2f best_q = synth.at<Vec2f>(p);
					float best_h = heightoffset.at<float>(p);
					float best_cost = numeric_limits<float>::infinity();

					for (int i = 0; i < 9; ++i) {
						Vec<Vec2f, 2> &candidates = candidates_full.at<Vec<Vec2f, 2>>(0, i);
						for (size_t k = 0; k < 2; ++k) {

							// the candidate pixel
							Vec2f q = candidates[k];

							// construct the neighbourhood
							Mat nq(1, 4, app_space.type());
							// for diagonal neighbours
							Mat nq_temp_synth_coords(1, 4, CV_32FC2);
							for (int n = 0; n < 4; ++n) {
								nq_temp_synth_coords.at<Vec2f>(0, n) = q + Vec2f(diagonal_delta[n].x, diagonal_delta[n].y);
							}
							remap(app_space, nq, nq_temp_synth_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);

							// the best height offset (TODO generic way to do this?)
							// accumulate an average height difference between the neighbourhoods
							float h = 0;
							for (int n = 0; n < 4; ++n)
								h += (np.at<Vec4f>(0, n)[0] - nq.at<Vec4f>(0, n)[0]);
							h /= 4;
							//h = 0; // uncomment for no height offset in the correction
							for (int n = 0; n < 4; ++n)
								nq.at<Vec4f>(0, n)[0] += h;

							// compute the cost (ssd of nq and np)
							float cost = norm(np, nq);

							// if not the the lowest k-value, double the cost
							//if (k > 0) cost *= 1.5;
							if (k > 0) cost *= 2;


							cout << "q" << q << endl;
							cout << "nq" << endl;
							cout << nq << endl;

							// find best correction
							if (cost < best_cost) {
								best_cost = cost;
								best_h = h;
								best_q = q;
							}
						}

					}


					cout << "best_q " << best_q  << endl;
					cout << "best_h " << best_h << endl;
					cout << "best_cost " << best_cost << endl;

					synth.at<Vec2f>(p) = best_q;
					heightoffset.at<float>(p) = best_h;
					//cost.at(p) = best_cost; // debug

				}
			}


		} // subpasses


		//// debug
		//auto height_img = arr_to_heightmap(resolve_coord(synthesis, example, height_offset));
		//auto error_img = arr_to_heatmap(cost_arr);

		//for (int x = 0; x < error_img.width(); x++) {
		//	for (int y = 0; y < error_img.height(); y++) {
		//		error_img.texel(x, y) = vec3(((height_img.texel(x, y) + 1) / 2).x) / 2 + error_img.texel(x, y) / 2;
		//	}
		//}

		//util::cache_upload(error_img, stringf("cost_arr pass=", pass, " | min=", cost_arr.min(), " : max=", cost_arr.max()));
	
	}




	void synthesizeDebug(Mat example, Mat app_space, Mat cohearance, Mat synth, Mat height, std::string tag) {
	
		Mat ind[4];
		split(app_space, ind);

		imwrite(util::stringf("output/appearance_", tag, "_0.png"), heightmapToImage(ind[0]));
		imwrite(util::stringf("output/appearance_", tag, "_1.png"), heightmapToImage(ind[1]));
		imwrite(util::stringf("output/appearance_", tag, "_2.png"), heightmapToImage(ind[2]));
		imwrite(util::stringf("output/appearance_", tag, "_3.png"), heightmapToImage(ind[3]));

		imwrite(util::stringf("output/example_", tag, ".png"), heightmapToImage(example));
		imwrite(util::stringf("output/syth_nnf_", tag, ".png"), nnfToImg(synth, example.size()));
		imwrite(util::stringf("output/height_", tag, ".png"), heightmapToImage(height));


		Mat nnf_img_k1(cohearance.rows, cohearance.cols, CV_8UC3, Scalar(0, 0, 0));
		Mat nnf_img_k2(cohearance.rows, cohearance.cols, CV_8UC3, Scalar(0, 0, 0));
		for (int r = 0; r < cohearance.rows; r++) {
			auto in_row = cohearance.ptr<Vec<Vec2f, 2>>(r);
			auto out_row_k1 = nnf_img_k1.ptr<Vec3b>(r);
			auto out_row_k2 = nnf_img_k2.ptr<Vec3b>(r);
			for (int c = 0; c < cohearance.cols; c++) {
				out_row_k1[c][2] = int((255.0 * in_row[c][0][0]) / app_space.cols); // cols -> r
				out_row_k1[c][1] = int((255.0 * in_row[c][0][1]) / app_space.rows); // rows -> g

				out_row_k2[c][2] = int((255.0 * in_row[c][1][0]) / app_space.cols); // cols -> r
				out_row_k2[c][1] = int((255.0 * in_row[c][1][1]) / app_space.rows); // rows -> g
			}
		}

		imwrite(util::stringf("output/NNF_", tag, "_k1.png"), nnf_img_k1);
		imwrite(util::stringf("output/NNF_", tag, "_k2.png"), nnf_img_k2);

		// recon
		Mat reconstructed;
		reconstructed.create(synth.size(), example.type());
		remap(example, reconstructed, synth, Mat(), INTER_LINEAR);
		reconstructed += height;
		imwrite(util::stringf("output/recon_", tag, ".png"), heightmapToImage(reconstructed));
	}


	Mat synthesizeTerrain(Mat example, Size synth_size, int example_levels, int synth_levels, string tag="") {

		assert(synth_levels > example_levels); // M > L

		// create exemplar pyramid
		vector<Mat> example_pyramid(synth_levels);
		buildPyramid(example, example_pyramid, synth_levels);

		// create synthesis pyramid
		Mat synth(synth_size.height, synth_size.width, CV_32FC2);
		Mat heightoffset(synth_size.height, synth_size.width, CV_32FC1, Scalar(0));
		vector<Mat> synth_pyramid(synth_levels);
		vector<Mat> heightoffset_pyramid(synth_levels);
		buildPyramid(synth, synth_pyramid, synth_levels);
		buildPyramid(heightoffset, heightoffset_pyramid, synth_levels);


		// initialization

		synth_pyramid[synth_levels - 1].setTo(Scalar(
			example_pyramid[synth_levels - 1].rows / 2.f,
			example_pyramid[synth_levels - 1].cols / 2.f
		));

		//randu(synth_pyramid[synth_levels - 1], Scalar(2, 2), Scalar(example_pyramid[synth_levels - 1].cols - 3, example_pyramid[synth_levels - 1].rows - 3));





		// iteration
		for (int level = synth_levels - 1; level-- > 0;) {

			// upsample / jitter
			pyrUpJitter(
				synth_pyramid[level + 1],
				heightoffset_pyramid[level + 1],
				synth_pyramid[level],
				heightoffset_pyramid[level],
				0.4
			);


			// TODO move enventually
			Mat app_space = appearance_space(example_pyramid[level]);
			Mat coherence = k2_patchmatch(app_space, app_space, 5, 4);



			// correction
			if (level < example_levels) {
				// TODO iterative correction
				for (int i = 0; i < 4; i++) { //hack for now
					correction(example_pyramid[level], app_space, coherence, synth_pyramid[level], heightoffset_pyramid[level]);
				}
			}


			{ // DEBUG //
				synthesizeDebug(
					example_pyramid[level],
					app_space,
					coherence,
					synth_pyramid[level],
					heightoffset_pyramid[level],
					util::stringf(level, tag)
				);
			} // DEBUG //

		}

		// reconstruct the synthesis
		Mat reconstructed;
		reconstructed.create(synth_size, example.type());
		remap(example, reconstructed, synth_pyramid[0], Mat(), INTER_LINEAR);
		reconstructed += heightoffset_pyramid[0];
		return reconstructed;
	}



	void test(Mat testimage) {


		// Test Gain
		//Mat synth = synthesizeTerrain(testimage, testimage.size(), 6, 7);
		
		terrain test_terrain = terrainReadTIFF("work/res/southern_alps_s045e169.tif");
		//for (int m = 1; m < 8; m++) {
		//	for (int l = m-1; l-->0; ) {
		//		Mat synth = synthesizeTerrain(test_terrain.heightmap, Size(512, 512), l, m, util::stringf("m",m,"l",l));
		//	}
		//}

		Mat synth = synthesizeTerrain(test_terrain.heightmap, Size(512, 512), 4, 7);
		//imwrite("output/syth.png", heightmapToImage(synth));




		//// Appearance-space
		//Mat data(testimage.size(), CV_32FC1);
		//testimage.convertTo(data, CV_32FC1);

		//Mat reduced = appearance_space(data);

		//Mat ind[4];
		//split(reduced, ind);

		//imwrite(util::stringf("output/appearance_0.png"), heightmapToImage(ind[0]));
		//imwrite(util::stringf("output/appearance_1.png"), heightmapToImage(ind[1]));
		//imwrite(util::stringf("output/appearance_2.png"), heightmapToImage(ind[2]));
		//imwrite(util::stringf("output/appearance_3.png"), heightmapToImage(ind[3]));




		//Mat terrain = reduced.clone();



		//// K=2 Patchmatch
		//Mat nnf = k2_patchmatch(testimage, testimage, 5, 4);

		//// convert offset coordinates to color image
		//Mat nnf_img_k1(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
		//Mat nnf_img_k2(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
		//for (int r = 0; r < nnf.rows; r++) {
		//	auto in_row = nnf.ptr<Vec<Vec2f, 2>>(r);
		//	auto out_row_k1 = nnf_img_k1.ptr<Vec3b>(r);
		//	auto out_row_k2 = nnf_img_k2.ptr<Vec3b>(r);
		//	for (int c = 0; c < nnf.cols; c++) {
		//		out_row_k1[c][2] = int((255.0 * in_row[c][0][0]) / testimage.cols); // cols -> r
		//		out_row_k1[c][1] = int((255.0 * in_row[c][0][1]) / testimage.rows); // rows -> g

		//		out_row_k2[c][2] = int((255.0 * in_row[c][1][0]) / testimage.cols); // cols -> r
		//		out_row_k2[c][1] = int((255.0 * in_row[c][1][1]) / testimage.rows); // rows -> g
		//	}
		//}

		//imwrite("output/NNF_k1.png", nnf_img_k1);
		//imwrite("output/NNF_k2.png", nnf_img_k2);




		//terrain test = terrainReadTIFF("work/res/southern_alps_s045e169.tif");
		//terrain test = terrainReadTIFF("work/res/mt_fuji_n035e138.tif"); 
		//imwrite("output/test_img1.png", heightmapToImage(test.heightmap));

	}

}