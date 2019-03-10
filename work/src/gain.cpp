
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
#include "gain.hpp"
#include "terrain.hpp"


using namespace cv;
using namespace std;



namespace {
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





	template<int AppComponents>
	Mat appearanceSpace(Mat data, const gain::synthesis_params &params) {
		assert(data.type() == CV_32FC1);

		Mat gaussian1d;
		getGaussianKernel(5, 1).convertTo(gaussian1d, CV_32FC1);
		Mat gaussian2d = gaussian1d * gaussian1d.t();

		// create the PCA set
		Mat neighbourhood_data(data.size(), CV_32FC(25));
		for (int i = 0; i < data.rows; i++) {
			for (int j = 0; j < data.cols; j++) {
				Point p(j, i);
				for (int ii = 0; ii < 5; ii++) {
					for (int jj = 0; jj < 5; jj++) {
						Point q(clamp(j + jj - 2, 0, data.cols - 1), clamp(i + ii - 2, 0, data.rows - 1));
						neighbourhood_data.at<Vec<Vec<float, 5>, 5>>(p)[ii][jj] = data.at<float>(q) * gaussian2d.at<float>(ii, jj);
					}
				}
			}
		}
		Mat pcaset = neighbourhood_data.reshape(1, data.cols * data.rows);

		// compute mean manually
		Mat pcaset_mean;
		reduce(pcaset, pcaset_mean, 1, CV_REDUCE_SUM);

		// compute PCA and transform to reduced eigen space
		PCA pca(pcaset, Mat(), PCA::DATA_AS_ROW, AppComponents);
		Mat reduced = pcaset * pca.eigenvectors.t();

		// EXTRA: scale the non-mean values of the apperance space appropriately
		if (params.scaleAppSpace) {
			Mat scaling_factor = pcaset_mean / reduced.col(0);
			for (int c = 0; c < reduced.cols; c++)
				reduced.col(c) = reduced.col(c).mul(scaling_factor);
		}

		// manually set the mean
		for (int i = 0; i < reduced.rows; i++) {
			reduced.at<float>(i, 0) = pcaset_mean.at<float>(i, 0);
		}

		// reshape and return
		Mat appspace = reduced.reshape(AppComponents, data.rows);
		return appspace;
	}




	float constriantCurve(float x, float d, const gain::synthesis_params &params) {
		float td = params.constraintScale * d;
		float bd = -params.constraintSlope / (3 * pow(d, 2));
		float omega;
		if (abs(x) > td) {
			float cd = bd * pow(td, 3) + params.constraintSlope * td;
			if (x < -td) {
				omega = params.constraintSlope * x + cd;
			}
			else {
				omega = params.constraintSlope * x - cd;
			}
		}
		else {
			omega = -bd * pow(x, 3);
		}
		return omega;
	}





	void correction(Mat example, Mat appSpace, Mat coherence, Mat synth, Mat heightoffset, const gain::synthesis_params &params) {
		assert(synth.type() == CV_32FC2);
		assert(heightoffset.type() == CV_32FC1);
		assert(example.size() == appSpace.size());
		assert(synth.size() == heightoffset.size());
		
		assert(appSpace.type() == CV_32FC4); // 4-channel appearance space means Vec4f
		assert(coherence.type() == CV_32FC4); // k=2 cohearance means Vec<Vec2f, 2>

		// delta set for diagonal neighbours
		const Point diagonalDelta[]{ {-1, -1}, { 1, -1}, {-1, 1}, {1, 1} };

		//  delta set for 3x3 neighbourhood
		const Point neighbourDelta[]{
			{ -1, -1 },{ -1, 0 },{ -1, 1 },
			{ 0, -1 }, { 0, 0 }, { 0, 1 },
			{ 1, -1 }, { 1, 0 }, { 1, 1 }
		};

		// blank matrix for constructing appearance space with height offset
		const Mat np_temp_zero = Mat::zeros(3, 4, CV_32FC1);

		// declare matrices (optimization for OpenCV)
		Mat np(1, 4, appSpace.type(), Scalar(0, 0, 0, 0));
		
		Mat np_temp_neighbour_coords(3, 4, CV_16SC2); // coordinates of neighbours and their adjacent offset neighbours
		Mat np_temp_synth_coords(3, 4, synth.type()); // values in synth retrieved via np_temp_neighbour_coords

		Mat np_temp_synth_offset(3, 4, synth.type()); // offset values from adjacent offset neighbours to back shift np_temp_synth_coords
		Mat np_temp_app(3, 4, appSpace.type()); // values in appSpace retrieved via np_temp_synth_coords
		Mat np_temp_ho_single(3, 4, heightoffset.type()); // values in heightoffset retrieved via np_temp_neighbour_coords
		Mat np_temp_ho_multi(3, 4, appSpace.type()); // space out values (h, 0, 0, 0) from np_temp_ho_single

		Mat candidates_neighbour_coords(1, 9, CV_16SC2); // coordinates of neighbours in synth
		Mat candidates_synth_coords(1, 9, synth.type()); // values in synth retrieved via candidates_neighbour_coords
		Mat candidates_neighbour_offset(1, 9, synth.type()); // the backward translation of neighbours in synth
		Mat candidates_full(1, 9, coherence.type()); // values in candidates retrieved via (candidates_synth_coords - candidates_neighbour_offset)

		Mat nq(1, 4, appSpace.type());
		Mat nq_temp_synth_coords(1, 4, CV_32FC2);

		// 4 subpasses
		const Point subpassDelta[]{ { 0,0 }, { 1,0 },{ 0,1 }, { 1,1 } };
		for (int subpass = 0; subpass < 4; ++subpass) {

			//#pragma omp parallel for
			for (int j = 0; j < synth.cols; j += 2) {
				for (int i = 0; i < synth.rows; i += 2) {

					// 1) Get the Pixel p, Neighbourhood np, and Height hp
					//
					Point p = Point(j, i) + subpassDelta[subpass];
					if (p.y >= synth.rows || p.x >= synth.cols) continue; // bound check

					// for every diagonal neighbour
					for (int n = 0; n < 4; ++n) {
						Point u = p + diagonalDelta[n];
						Point vv[]{ // adjacent location offset (0,0), (+-1,0), (0,+-1)
							{ 0, 0 },
							{ diagonalDelta[n].x, 0 },
							{ 0, diagonalDelta[n].y }
						};

						// 3x4 matrix, each column is a diagonal neighbour and its adjacent offset neighbours
						for (int nn = 0; nn < 3; nn++) {
							Point temp = u + vv[nn];
							np_temp_neighbour_coords.at<Vec2s>(nn, n) = Vec2s(temp.x, temp.y);
							np_temp_synth_offset.at<Vec2f>(nn, n) = Vec2f(vv[nn].x, vv[nn].y);
						}
					}

					// fill out the temp matricies
					remap(synth, np_temp_synth_coords, np_temp_neighbour_coords, Mat(), INTER_NEAREST, BORDER_REPLICATE);
					np_temp_synth_coords -= np_temp_synth_offset;
					remap(appSpace, np_temp_app, np_temp_synth_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);
					remap(heightoffset, np_temp_ho_single, np_temp_neighbour_coords, Mat(), INTER_NEAREST, BORDER_REPLICATE);
					merge(vector<Mat>{ np_temp_ho_single, np_temp_zero, np_temp_zero, np_temp_zero }, np_temp_ho_multi);

					// average appearance space combine with height offset 
					np_temp_app += np_temp_ho_multi;
					reduce(np_temp_app, np, 0, CV_REDUCE_AVG);

					// height
					float hp = 0;
					for (int n = 0; n < 4; ++n) {
						hp += np.at<Vec4f>(0, n)[0];
					}
					hp /= 4;



					// 2) Find the best matching Pixel/Neighbourhood
					//

					// collect candidates of neighbours
					for (int i = 0; i < 9; ++i) {
						Point temp = clampToMat(p + neighbourDelta[i], synth);
						candidates_neighbour_coords.at<Vec2s>(0, i) = Vec2s(temp.x, temp.y);
						candidates_neighbour_offset.at<Vec2f>(0, i) = Vec2f(neighbourDelta[i].x, neighbourDelta[i].y);
					}

					// fill out the candidate matricies
					remap(synth, candidates_synth_coords, candidates_neighbour_coords, Mat(), INTER_NEAREST, BORDER_REPLICATE);
					candidates_synth_coords -= candidates_neighbour_offset;
					remap(coherence, candidates_full, candidates_synth_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);

					// current best
					Vec2f best_q = synth.at<Vec2f>(p);
					float best_h = heightoffset.at<float>(p);
					float bestCost = numeric_limits<float>::infinity();

					// find the best neighbour candidate
					for (int i = 0; i < 9; ++i) {

						// for both k=2 candidates
						for (size_t k = 0; k < 2; ++k) {

							// the candidate exampleCoord
							Vec2f q = candidates_full.at<Vec<Vec2f, 2>>(0, i)[k];

							// create neighbourhood from diagonal neighbours in the example
							for (int n = 0; n < 4; ++n) {
								nq_temp_synth_coords.at<Vec2f>(0, n) = q + Vec2f(diagonalDelta[n].x, diagonalDelta[n].y);
							}
							remap(appSpace, nq, nq_temp_synth_coords, Mat(), INTER_LINEAR, BORDER_REPLICATE);

							// calculate a height offset that minimizes the SSD between np and nq
							// accumulate an average height difference between the neighbourhoods
							float hq = 0;
							for (int n = 0; n < 4; ++n) {
								hq += nq.at<Vec4f>(0, n)[0];
							}
							hq /= 4;
							float h = hp - hq;
							//h = 0; // uncomment for no height offset in the correction

							// set the height offset in the Matrix
							for (int n = 0; n < 4; ++n) {
								nq.at<Vec4f>(0, n)[0] += h;
							}

							// compute the cost
							float cost = norm(np, nq);
							if (k > 0) cost *= 2; // if not the the lowest k-value, double the cost
							if (cost < bestCost) {
								bestCost = cost;
								best_q = q;
								best_h = h;
							}
						}
					}



					// 3) Adjust height-offset with constraints
					// 

					//float target_h;
					//float target_h;


					//for (auto c : params.constraints) {

					//}




					// 4) Set the value
					//
					synth.at<Vec2f>(p) = best_q;
					heightoffset.at<float>(p) = best_h;
				}
			}


		} // subpasses

	}



	void synthesizeDebug(Mat example, Mat app_space, Mat cohearance, Mat synth, Mat height, std::string tag) {
		using namespace gain;

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
}





namespace gain {

	Mat synthesizeTerrain(Mat example, synthesis_params params) {
		assert(synth_levels > example_levels); // M > L

		// 1) Initialization
		//

		// determinism
		util::reset_random();

		// create exemplar pyramid
		vector<Mat> example_pyramid(params.synthesisLevels);
		buildPyramid(example, example_pyramid, params.synthesisLevels);

		// create synthesis pyramid
		Mat synth(params.synthesisSize.height, params.synthesisSize.width, CV_32FC2);
		Mat heightoffset(params.synthesisSize.height, params.synthesisSize.width, CV_32FC1, Scalar(0));
		vector<Mat> synth_pyramid(params.synthesisLevels);
		vector<Mat> heightoffset_pyramid(params.synthesisLevels);
		buildPyramid(synth, synth_pyramid, params.synthesisLevels);
		buildPyramid(heightoffset, heightoffset_pyramid, params.synthesisLevels);


		// center initialization
		synth_pyramid[params.synthesisLevels - 1].setTo(Scalar(
			example_pyramid[params.synthesisLevels - 1].rows / 2.f,
			example_pyramid[params.synthesisLevels - 1].cols / 2.f
		));

		// random initialization
		if (params.randomInit) {
			for (int i = 0; i < synth_pyramid[params.synthesisLevels - 1].rows; ++i) {
				for (int j = 0; j < synth_pyramid[params.synthesisLevels - 1].cols; ++j) {
					synth_pyramid[params.synthesisLevels - 1].at<Vec2f>(i, j) = Vec2f(
						util::random<float>(2, example_pyramid[params.synthesisLevels - 1].cols - 2),
						util::random<float>(2, example_pyramid[params.synthesisLevels - 1].rows - 2)
					);
				}
			}
			//randu(synth_pyramid[synth_levels - 1], Scalar(2, 2), Scalar(example_pyramid[synth_levels - 1].cols - 2, example_pyramid[synth_levels - 1].rows - 2));
		}

		// iteration
		for (int level = params.synthesisLevels - 1; level-- > 0;) {

			// 2) upsample / jitter
			//
			pyrUpJitter(
				synth_pyramid[level + 1],
				heightoffset_pyramid[level + 1],
				synth_pyramid[level],
				heightoffset_pyramid[level],
				params.jitter
			);

			Mat app_space = appearanceSpace<4>(example_pyramid[level], params);
			Mat coherence = k2Patchmatch(app_space, app_space, 5, 4);



			// 3) Correction
			//
			if (level < params.exampleLevels) {
				for (int i = 0; i < params.correctionIter; ++i) {
					correction(example_pyramid[level], app_space, coherence, synth_pyramid[level], heightoffset_pyramid[level], params);
				}
			}


			{ // DEBUG //
				synthesizeDebug(
					example_pyramid[level],
					app_space,
					coherence,
					synth_pyramid[level],
					heightoffset_pyramid[level],
					util::stringf(level, "")
				);
			} // DEBUG //

		}

		// reconstruct the synthesis
		Mat reconstructed;
		reconstructed.create(params.synthesisSize, example.type());
		remap(example, reconstructed, synth_pyramid[0], Mat(), INTER_LINEAR);
		reconstructed += heightoffset_pyramid[0];
		return reconstructed;
	}
}