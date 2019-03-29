
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




	void correction(Mat example, Mat appSpace, Mat coherence, Mat synth, Mat heightoffset, Mat constraints, Mat cHeight, Mat cDistance, const gain::synthesis_params &params) {
		assert(example.type() == CV_32FC1);
		assert(appSpace.type() == CV_32FC4); // 4-channel appearance space means Vec4f
		assert(coherence.type() == CV_32FC4); // k=2 cohearance means Vec<Vec2f, 2>
		assert(synth.type() == CV_32FC2);
		assert(heightoffset.type() == CV_32FC1);
		assert(constraints.type() == CV_8UC1);
		assert(cHeight.type() == CV_32FC1);
		assert(cDistance.type() == CV_32FC1);

		assert(example.size() == appSpace.size());
		assert(synth.size() == heightoffset.size());
		assert(synth.size() == constraints.size());
		assert(synth.size() == cHeight.size());
		assert(synth.size() == cDistance.size());
		

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


		// Correction iteration
		// 4 subpasses
		//
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


					// 2) Constraints
					//
					float hp = 0; // average neighbourhood height
					for (int n = 0; n < 4; ++n) {

						//TODO just using the center of the nighbourhood for working out the correct constraints
						// maybe it should be done for all diag neighbours
						// maybe it should be donw for the adjenct neighbours to diag neighbuors as well
						// seriously... who the fuck knows...
						if (constraints.at<bool>(p)) {
							float hc = np.at<Vec4f>(0, n)[0];
							float ht = cHeight.at<float>(p);
							float x = ht - hc;
							float d = cDistance.at<float>(i, j);
							float sd = params.constraintScale * d;

							float theta;
							if (x < -sd) {
								theta = params.constraintSlope * x + (2 * params.constraintSlope * sd) / 3;
							}
							else if (x > sd) {
								theta = params.constraintSlope * x - (2 * params.constraintSlope * sd) / 3;
							}
							else {
								theta = (params.constraintSlope * pow(x, 3)) / (3 * pow(params.constraintScale, 2), pow(d, 2));
							}

							float newHeight = hc + theta;
							np.at<Vec4f>(0, n)[0] = newHeight;
						}
						hp += np.at<Vec4f>(0, n)[0];
					}
					hp /= 4;
					



					// 3) Find the best matching Pixel/Neighbourhood
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
							hq /= 4; // calculate the average height of the candidate neighbourhood
							float h = hp - hq; // what we add to the candidate to bring it up to the Neighbourhood np
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

					// 4) Set the value
					//
					synth.at<Vec2f>(p) = best_q;
					heightoffset.at<float>(p) = best_h;
				}
			}


		} // subpasses

	}




	Mat reconstruct(Mat example, Mat synth, Mat heightoffset) {
		Mat reconstructed;
		remap(example, reconstructed, synth, Mat(), INTER_LINEAR);
		reconstructed += heightoffset;
		return reconstructed;
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


	constraint_value point_constraint::calculateValue(cv::Vec2f pos, float maxRadius) {
		constraint_value v;
		v.distance = norm(pos - position);
		v.height = height + (pos - position).dot(gradient);
		v.valid = v.distance <= min(radius, maxRadius);
		return v;
	}

	constraint_value curve_constraint::calculateValue(cv::Vec2f pos, float maxRadius) {
		// first value
		constraint_value v;
		v.distance = norm(path[0] - pos);
		v.height = curveNodes[0].height;
		float side = (pos[0] - path[0][0]) * (path[1][1] - path[0][1]) - (pos[1] - path[0][1]) * (path[1][0] - path[0][0]);
		v.height += v.distance * (side > 0) ? curveNodes[0].leftGradient : curveNodes[0].rightGradient;

		// for every edge between the nodes
		int headNode = 0;
		for (int i = 1; i < path.size(); i++) {

			// move to the next "segment"
			if (headNode + 2 < curveNodes.size() && i == curveNodes[headNode + 1].index) {
				headNode++;
			}

			// get points along the segment
			Vec2f p1 = path[i - 1];
			Vec2f p2 = path[i];
			float d = norm(p2 - pos); // distance to constraint
			float t = (i - curveNodes[headNode].index) / float(curveNodes[headNode + 1].index - curveNodes[headNode].index); // [0, 1] between the two curve nodes

			if (d < v.distance) {
				v.distance = d;
				float h1 = curveNodes[headNode].height;
				float h2 = curveNodes[headNode + 1].height;
				v.height = ((1 - t) * h1 + t * h2);

				// calculate the side (+/-)
				// if p1 is north, p2 is south, then right (east) is positive, and left is negative
				side = (pos[0] - p1[0]) * (p2[1] - p1[1]) - (pos[1] - p1[1]) * (p2[0] - p1[0]);

				float g1, g2, r1, r2;
				if (side > 0) { // right
					g1 = curveNodes[headNode].rightGradient;
					g2 = curveNodes[headNode + 1].rightGradient;
					r1 = curveNodes[headNode].rightRadius;
					r2 = curveNodes[headNode + 1].rightRadius;
				}
				else { // left
					g1 = curveNodes[headNode].leftGradient;
					g2 = curveNodes[headNode + 1].leftGradient;
					r1 = curveNodes[headNode].leftRadius;
					r2 = curveNodes[headNode + 1].leftRadius;
				}
				v.height += v.distance * ((1 - t) * g1 + t * g2);
				v.valid = v.distance < ((1 - t) * r1 + t * r2);
			}

		}

		// check if valid
		v.valid = v.distance <= maxRadius;

		return v;
	}


	Mat synthesizeTerrain(Mat example, synthesis_params params) {
		assert(params.synthesisLevels > params.exampleLevels); // M > L

		// 1) Initialization
		//

		// determinism
		util::reset_random();

		// create exemplar pyramid
		vector<Mat> examplePyramid(params.synthesisLevels);
		buildPyramid(example, examplePyramid, params.synthesisLevels);

		// create synthesis pyramid
		Mat synth(params.synthesisSize.height, params.synthesisSize.width, CV_32FC2);
		Mat heightoffset(params.synthesisSize.height, params.synthesisSize.width, CV_32FC1, Scalar(0));
		vector<Mat> synthPyramid(params.synthesisLevels);
		vector<Mat> heightoffsetPyramid(params.synthesisLevels);
		buildPyramid(synth, synthPyramid, params.synthesisLevels);
		buildPyramid(heightoffset, heightoffsetPyramid, params.synthesisLevels);

		// center initialization
		synthPyramid[params.synthesisLevels - 1].setTo(Scalar(
			examplePyramid[params.synthesisLevels - 1].rows / 2.f,
			examplePyramid[params.synthesisLevels - 1].cols / 2.f
		));

		// random initialization
		if (params.randomInit) {
			for (int i = 0; i < synthPyramid[params.synthesisLevels - 1].rows; ++i) {
				for (int j = 0; j < synthPyramid[params.synthesisLevels - 1].cols; ++j) {
					synthPyramid[params.synthesisLevels - 1].at<Vec2f>(i, j) = Vec2f(
						util::random<float>(2, examplePyramid[params.synthesisLevels - 1].cols - 2),
						util::random<float>(2, examplePyramid[params.synthesisLevels - 1].rows - 2)
					);
				}
			}
			//randu(synthPyramid[synth_levels - 1], Scalar(2, 2), Scalar(examplePyramid[synth_levels - 1].cols - 2, examplePyramid[synth_levels - 1].rows - 2));
		}

		// pre-calcuate parameters for constraints
		Mat constraintMask(synth.rows, synth.cols, CV_8UC1, Scalar(0));
		Mat constraintHeightMap(synth.rows, synth.cols, CV_32FC1, Scalar(0));
		Mat constraintDistanceMap(synth.rows, synth.cols, CV_32FC1, Scalar(0));

		// create constraint pyramid
		vector<Mat> constraintMaskPyramid(params.exampleLevels);
		vector<Mat> constraintHeightPyramid(params.exampleLevels);
		vector<Mat> constraintDistancePyramid(params.exampleLevels);
		buildPyramid(constraintMask, constraintMaskPyramid, params.exampleLevels);
		buildPyramid(constraintHeightMap, constraintHeightPyramid, params.exampleLevels);
		buildPyramid(constraintDistanceMap, constraintDistancePyramid, params.exampleLevels);

		// only compute for all but the finer 2 levels of the synthesis
		for (int level = params.exampleLevels - 1; level-- > 2;) {
			constraintMask = constraintMaskPyramid[level];
			constraintHeightMap = constraintHeightPyramid[level];
			constraintDistanceMap = constraintDistancePyramid[level];

			for (int j = 0; j < constraintMask.cols; j++) {
				for (int i = 0; i < constraintMask.rows; i++) {
					float maxRad = 4 * pow(2, level);
					Vec2f p(j+0.5, i+0.5);
					p *= pow(2, level); // convert to same coord system as constraints

					// sum of constraint values
					int count = 0;
					float distance = 0;
					float weightSum = 0;
					float targetHeight = 0;

					// aggregate constraints
					vector<constraint_value> values;
					for (point_constraint c : params.pointConstraints) {
						values.push_back(c.calculateValue(p, maxRad));
					}
					for (curve_constraint c : params.curveConstraints) {
						values.push_back(c.calculateValue(p, maxRad));
					}

					for (constraint_value value : values) {
						if (!value.valid) continue;
						float weight = 1 / value.distance;

						// case where point is on a constraint
						if (isinf(weight) || isnan(weight)) {
							count = 1;
							weightSum = 1;
							targetHeight = value.height;
							distance = value.distance;
							break;
						}

						count++;
						weightSum += weight;
						targetHeight += weight * value.height;
						distance += weight * value.distance;
					}

					if (count > 0) {
						constraintMask.at<bool>(i, j) = true;
						constraintHeightMap.at<float>(i, j) = targetHeight / weightSum;
						constraintDistanceMap.at<float>(i, j) = distance / weightSum;
					}
				}
			}

			imwrite(util::stringf("output/constraintMask", level, ".png"), constraintMask);
			imwrite(util::stringf("output/constraintHeightMap", level, ".png"), heightmapToImage(constraintHeightMap));
			imwrite(util::stringf("output/constraintDistanceMap", level, ".png"), heightmapToImage(constraintDistanceMap));
		}


		//return Mat();


		// iteration
		for (int level = params.synthesisLevels - 1; level-- > 0;) {

			// 2) upsample / jitter
			//
			pyrUpJitter(
				synthPyramid[level + 1],
				heightoffsetPyramid[level + 1],
				synthPyramid[level],
				heightoffsetPyramid[level],
				params.jitter
			);

			Mat appSpace = appearanceSpace<4>(examplePyramid[level], params);
			Mat coherence = k2Patchmatch(appSpace, appSpace, 5, 4);



			// 3) Correction
			//
			if (level < params.exampleLevels) {
				for (int i = 0; i < params.correctionIter; ++i) {
					correction(
						examplePyramid[level],
						appSpace,
						coherence,
						synthPyramid[level],
						heightoffsetPyramid[level],
						constraintMaskPyramid[level],
						constraintHeightPyramid[level],
						constraintDistancePyramid[level],
						params
					);
				}
			}


			{ // DEBUG //
				synthesizeDebug(
					examplePyramid[level],
					appSpace,
					coherence,
					synthPyramid[level],
					heightoffsetPyramid[level],
					util::stringf(level, "")
				);
			} // DEBUG //

		}

		// reconstruct the synthesis
		Mat reconstructed;
		reconstructed.create(params.synthesisSize, example.type());
		remap(example, reconstructed, synthPyramid[0], Mat(), INTER_LINEAR);
		reconstructed += heightoffsetPyramid[0];
		return reconstructed;
	}
}