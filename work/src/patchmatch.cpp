
// std
#include <limits>
#include <iostream>

// project
#include "patchmatch.hpp"

using namespace cv;
using namespace std;

namespace {
	// requires that the target is buffered images (with boarders equal to patch_size)
	void improve_nnf(Mat target, Point p, Mat source, Point q, Mat nnf, Mat cost, int patch_size) {
		int hp1 = patch_size / 2;
		int hp2 = patch_size - hp1;

		// get patches
		auto patch_p = target(Range(p.y - hp1, p.y + hp2) + patch_size, Range(p.x - hp1, p.x + hp2) + patch_size);
		auto patch_q = source(Range(q.y - hp1, q.y + hp2), Range(q.x - hp1, q.x + hp2));

		// cost (ssd)
		float c = norm(patch_p, patch_q);

		if (c < cost.at<float>(p)) {
			nnf.at<Point>(p) = q;
			cost.at<float>(p) = c;
		}
	}

	Point clamp_rect(const Point& p, const Rect& rect) {
		return Point(max(rect.x, min(p.x, rect.x + rect.width)), max(rect.y, min(p.y, rect.y + rect.height)));
	}
}

// returns a matrix of Points representing the best absolute position
Mat patchmatch(Mat source, Mat target, int patch_size, float iterations, cv::Mat est) {
	assert(source.type() == target.type());

	int hp = patch_size / 2;

	// Initalize
	//
	// nnf and cost matrices
	Mat cost(target.rows, target.cols, CV_32FC1, Scalar(std::numeric_limits<float>::infinity())); // 1-channel float
	Mat nnf(target.rows, target.cols, CV_32SC2); // 2-channel 32-bit signed-int matrix
	if (est.size() == nnf.size() && est.type() == nnf.type()) nnf = est.clone();
	else randu(nnf, Scalar(hp, hp), Scalar(source.cols - hp - 1, source.rows - hp - 1));
	RNG rng(9001); // opencv rng

	// edge buffered images (for improve_nnf only)
	Mat target_buffer;
	copyMakeBorder(target, target_buffer, patch_size, patch_size, patch_size, patch_size, BORDER_REPLICATE);

	// for contains operations
	Rect source_rect(Point(hp, hp), source.size() - Size2i(patch_size, patch_size));
	Rect target_rect(Point(), target.size());


	// calculate the correct cost for nnf init
	for (int i = 0; i < nnf.rows; i++) {
		for (int j = 0; j < nnf.cols; j++) {
			improve_nnf(target_buffer, Point(j, i), source, nnf.at<Point>(i, j), nnf, cost, patch_size);
		}
	}


	// Iteration
	//
	for (int iter = 0; iter < iterations; iter++) {

		// Propagation
		//
		for (int i = 0; i < nnf.rows; i++) {
			for (int j = 0; j < nnf.cols; j++) {
				int s = (iter % 2 == 0) ? 1 : -1;
				Point p = (iter % 2 == 0) ? Point(j, i) : Point(nnf.cols - j - 1, nnf.rows - i - 1);
				Point q1 = p - Point(0, s);
				Point q2 = p - Point(s, 0);

				if (target_rect.contains(q1)) {
					Point cand = nnf.at<Point>(q1) + Point(0, s);
					//if (source_rect.contains(cand)) {
					//	improve_nnf(target_buffer, p, source, cand, nnf, cost, patch_size);
					//}
					improve_nnf(target_buffer, p, source, clamp_rect(cand, source_rect), nnf, cost, patch_size);
				}

				if (target_rect.contains(q2)) {
					Point cand = nnf.at<Point>(q2) + Point(s, 0);
					//if (source_rect.contains(cand)) {
					//	improve_nnf(target_buffer, p, source, cand, nnf, cost, patch_size);
					//}
					improve_nnf(target_buffer, p, source, clamp_rect(cand, source_rect), nnf, cost, patch_size);
				}
			}
		}

		// Random Search
		//
		for (int i = 0; i < nnf.rows; i++) {
			for (int j = 0; j < nnf.cols; j++) {
				Point p(j, i);

				// ever decreasing window for random search
				for (float t = max(source.rows, source.cols); t >= 1; t /= 2) {

					// random point centered around current best point
					Point r = nnf.at<Point>(p) + Point(rng.uniform(-t, t), rng.uniform(-t, t));
					//if (source_rect.contains(r)) {
					//	improve_nnf(target_buffer, p, source, r, nnf, cost, patch_size);
					//}
					improve_nnf(target_buffer, p, source, clamp_rect(r, source_rect), nnf, cost, patch_size);
				}
			}
		}

	} // end iter


	return nnf;
}



Mat reconstruct(Mat source, Mat nnf, int patch_size) {

	// create output buffers
	Mat output_buffer(nnf.size() + Size2i(patch_size * 2, patch_size * 2), CV_32FC3, Scalar(0, 0, 0));
	Mat output_count(output_buffer.size(), CV_32FC3, Scalar(0, 0, 0));

	// convert source to same type
	Mat source_buffer(source.size(), CV_32FC3);
	source.convertTo(source_buffer, CV_32FC3);
	copyMakeBorder(source_buffer, source_buffer, patch_size, patch_size, patch_size, patch_size, BORDER_CONSTANT);

	// calculate patch sizes
	int hp1 = patch_size / 2;
	int hp2 = patch_size - hp1;

	for (int i = 0; i < nnf.rows; i++) {
		for (int j = 0; j < nnf.cols; j++) {
			Point p(j, i);
			Point q = nnf.at<Point>(i, j);

			auto patch_p = output_buffer(Range(p.y - hp1, p.y + hp2) + patch_size, Range(p.x - hp1, p.x + hp2) + patch_size);
			auto patch_q = source_buffer(Range(q.y - hp1, q.y + hp2) + patch_size, Range(q.x - hp1, q.x + hp2) + patch_size);
			auto patch_count = output_count(Range(p.y - hp1, p.y + hp2) + patch_size, Range(p.x - hp1, p.x + hp2) + patch_size);

			patch_p += patch_q;
			//patch_count += 1;
			add(patch_count, Scalar(1, 1, 1), patch_count);
		}
	}

	//output_buffer /= output_count;
	divide(output_buffer, output_count, output_buffer);

	Mat output(nnf.size(), source.type());
	output_buffer(Range(patch_size, output.rows + patch_size), Range(patch_size, output.cols + patch_size)).convertTo(output, source.type());

	return output;
}