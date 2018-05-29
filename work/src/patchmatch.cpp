
// std
#include <limits>
#include <iostream>

// project
#include "patchmatch.hpp"
#include "opencv_util.hpp"

// opencv
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


namespace {

	void improve_k2_nnf(Point p, Vec2f q, float q_cost, Mat nnf, Mat cost, float min_dis) {
		if (q_cost < cost.at<Vec2f>(p)[0]) { // better than the first result
			Vec<Vec2f, 2> temp = nnf.at<Vec<Vec2f, 2>>(p);
			Vec2f temp_cost= cost.at<Vec2f>(p);

			// set the best result and second best as temp
			nnf.at<Vec<Vec2f, 2>>(p) = Vec<Vec2f, 2>{ q, q };
			cost.at<Vec2f>(p) = Vec2f{ q_cost, numeric_limits<float>::infinity() };

			// try to push the best results again
			improve_k2_nnf(p, temp[0], temp_cost[0], nnf, cost, min_dis);
			improve_k2_nnf(p, temp[1], temp_cost[1], nnf, cost, min_dis);
		}
		else if (q_cost < cost.at<Vec2f>(p)[1] && norm(q, nnf.at<Vec<Vec2f, 2>>(p)[0]) > min_dis) { // better than second result and far from first result
			nnf.at<Vec<Vec2f, 2>>(p)[1] = q;
			cost.at<Vec2f>(p)[1] = q_cost;
		}
	}


	// requires that the target is buffered images (with borders equal to patch_size)
	void improve_k2_nnf(Mat target, Point p, Mat source, Vec2f q, Mat nnf, Mat cost, int patch_size, float min_dis) {
		int hp1 = patch_size / 2;
		int hp2 = patch_size - hp1;
		
		// get patches
		auto patch_p = target(Range(p.y - hp1, p.y + hp2) + patch_size, Range(p.x - hp1, p.x + hp2) + patch_size);
		Mat patch_q(patch_size, patch_size, source.type());
		Mat map1(patch_size, patch_size, CV_32FC2);
		for (int i = 0; i < patch_size; i++)
			for (int j = 0; j < patch_size; j++)
				map1.at<Vec2f>(Point(j, i)) = q + Vec2f(j - hp1, i - hp1);
		remap(source, patch_q, map1, Mat(), INTER_LINEAR, BORDER_REPLICATE);
				
		// cost (ssd)
		float c = norm(patch_p, patch_q);

		// recursive replace
		improve_k2_nnf(p, q, c, nnf, cost, min_dis);
	}


	Vec2f clampToRect(const Vec2f& p, const Rect& rect) {
		return Vec2f(clamp<float>(p[0], rect.x, rect.x + rect.width - 1), clamp<float>(p[1], rect.y, rect.y + rect.height - 1));
	}
}


// returns a matrix of Points representing the best absolute position
Mat k2_patchmatch(Mat source, Mat target, int patch_size, float iterations, cv::Mat est) {
	assert(source.type() == target.type());

	int hp = patch_size / 2;
	float min_dis = norm(Vec2f(source.cols, source.rows)) * 0.05;

	// Initalize
	//
	// nnf and cost matrices
	RNG rng(9001); // opencv rng
	Mat cost(target.rows, target.cols, CV_32FC2, Scalar(numeric_limits<float>::infinity(), numeric_limits<float>::infinity())); // 2-channel float
	Mat nnf(target.rows, target.cols, CV_32FC4); // (Vec<Vec2f, 2>) 4-channel float matrix
	if (est.size() == nnf.size() && est.type() == nnf.type()) nnf = est.clone();
	else {
		for (int i = 0; i < nnf.rows; i++) {
			auto row = nnf.ptr<Vec<Vec2f, 2>>(i);
			for (int j = 0; j < nnf.cols; j++) {
				row[j][0] = Vec2f(rng.uniform(hp, source.cols - hp - 1), rng.uniform(hp, source.rows - hp - 1));
				row[j][1] = Vec2f(rng.uniform(hp, source.cols - hp - 1), rng.uniform(hp, source.rows - hp - 1));
			}
		}
	}

	// edge buffered images (for improve_nnf only)
	Mat target_buffer;
	copyMakeBorder(target, target_buffer, patch_size, patch_size, patch_size, patch_size, BORDER_REPLICATE);

	// for contains operations
	Rect source_rect(Point(hp, hp), source.size() - Size2i(patch_size, patch_size));
	Rect target_rect(Point(), target.size());


	// calculate the correct cost for nnf init
	for (int i = 0; i < nnf.rows; i++) {
		for (int j = 0; j < nnf.cols; j++) {
			Vec<Vec2f, 2> temp = nnf.at<Vec<Vec2f, 2>>(Point(j, i));
			improve_k2_nnf(target_buffer, Point(j, i), source, temp[0], nnf, cost, patch_size, min_dis);
			improve_k2_nnf(target_buffer, Point(j, i), source, temp[1], nnf, cost, patch_size, min_dis);
		}
	}


	// Iteration
	//
	for (int iter = 0; iter < iterations; iter++) {
		cout << iter << endl;

		cout << "prop" << endl;
		// Propagation
		//
		for (int i = 0; i < nnf.rows; i++) {
			for (int j = 0; j < nnf.cols; j++) {
				int s = (iter % 2 == 0) ? 1 : -1;
				Point p = (iter % 2 == 0) ? Point(j, i) : Point(nnf.cols - j - 1, nnf.rows - i - 1);
				Point q1 = p - Point(0, s); // neighbour 1
				Point q2 = p - Point(s, 0); // neighbour 2

				for (int k = 0; k < 2; k++) {
					if (target_rect.contains(q1)) {
						Vec2f cand = nnf.at<Vec<Vec2f, 2>>(q1)[k] + Vec2f(0, s);
						improve_k2_nnf(target_buffer, p, source, clampToRect(cand, source_rect), nnf, cost, patch_size, min_dis);
					}
					if (target_rect.contains(q2)) {
						Vec2f cand = nnf.at<Vec<Vec2f, 2>>(q2)[k] + Vec2f(s, 0);
						improve_k2_nnf(target_buffer, p, source, clampToRect(cand, source_rect), nnf, cost, patch_size, min_dis);
					}
				}
			}
		}

		cout << "rand" << endl;
		// Random Search
		//
		for (int i = 0; i < nnf.rows; i++) {
			for (int j = 0; j < nnf.cols; j++) {
				Point p(j, i);

				for (int k = 0; k < 2; k++) {
					// ever decreasing window for random search
					for (float t = max(source.rows, source.cols); t >= 1; t /= 2) {

						// random Point centered around current best Point
						Vec2f r = nnf.at<Vec<Vec2f, 2>>(p)[k] + Vec2f(rng.uniform(-t, t), rng.uniform(-t, t));
						improve_k2_nnf(target_buffer, p, source, clampToRect(r, source_rect), nnf, cost, patch_size, min_dis);
					}
				}
			}
		}

	} // end iter


	return nnf;
}


// TODO
Mat reconstruct(Mat source, Mat nnf, int patch_size) {

	//// create output buffers
	//Mat output_buffer(nnf.size() + Size2i(patch_size * 2, patch_size * 2), CV_32FC3, Scalar(0, 0, 0));
	//Mat output_count(output_buffer.size(), CV_32FC3, Scalar(0, 0, 0));

	//// convert source to same type
	//Mat source_buffer(source.size(), CV_32FC3);
	//source.convertTo(source_buffer, CV_32FC3);
	//copyMakeBorder(source_buffer, source_buffer, patch_size, patch_size, patch_size, patch_size, BORDER_CONSTANT);

	//// calculate patch sizes
	//int hp1 = patch_size / 2;
	//int hp2 = patch_size - hp1;

	//for (int i = 0; i < nnf.rows; i++) {
	//	for (int j = 0; j < nnf.cols; j++) {
	//		Point p(j, i);
	//		Point q = nnf.at<Point>(i, j);

	//		auto patch_p = output_buffer(Range(p.y - hp1, p.y + hp2) + patch_size, Range(p.x - hp1, p.x + hp2) + patch_size);
	//		auto patch_q = source_buffer(Range(q.y - hp1, q.y + hp2) + patch_size, Range(q.x - hp1, q.x + hp2) + patch_size);
	//		auto patch_count = output_count(Range(p.y - hp1, p.y + hp2) + patch_size, Range(p.x - hp1, p.x + hp2) + patch_size);

	//		patch_p += patch_q;
	//		//patch_count += 1;
	//		add(patch_count, Scalar(1, 1, 1), patch_count);
	//	}
	//}

	////output_buffer /= output_count;
	//divide(output_buffer, output_count, output_buffer);

	Mat output(nnf.size(), source.type());
	//output_buffer(Range(patch_size, output.rows + patch_size), Range(patch_size, output.cols + patch_size)).convertTo(output, source.type());

	return output;
}

//TODO
// convert offset coordinates to color image
cv::Mat nnfToImg(cv::Mat nnf, cv::Size s, bool absolute) {
	assert(nnf.type() == CV_32FC2);

	cv::Mat nnf_img(nnf.rows, nnf.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Rect rect(cv::Point(0, 0), s);

	for (int r = 0; r < nnf.rows; r++) {
		auto in_row = nnf.ptr<cv::Vec2f>(r);
		auto out_row = nnf_img.ptr<cv::Vec3b>(r);
		for (int c = 0; c < nnf.cols; c++) {
			float j = absolute ? in_row[c][0] : in_row[c][0] + c;
			float i = absolute ? in_row[c][1] : in_row[c][1] + r;
			if (!rect.contains(cv::Point(j, i))) {
				/* coordinate is outside the boundry, insert error of choice */
				std::cerr << "AAAAHHHHHHHHHH " << (absolute ? in_row[c][1] : in_row[c][1] + r) << std::endl;
			}
			out_row[c][2] = int(j * 255 / s.width);  // cols -> red
			out_row[c][1] = int(i * 255 / s.height); // rows -> green
			out_row[c][0] = 0;// 255 - max(out_row[c][2], out_row[c][1]);
		}
	}

	return nnf_img;
}


