#pragma once

// opencv
#include <opencv2/core.hpp>

inline cv::Point clampToRect(const cv::Point& p, const cv::Rect& rect) {
	using namespace cv;
	using namespace std;
	return Point(max(rect.x, min(p.x, rect.x + rect.width)), max(rect.y, min(p.y, rect.y + rect.height)));
}

inline cv::Point clampToMat(const cv::Point& p, const cv::Mat& mat) {
	using namespace cv;
	using namespace std;
	return Point(max(0, min(p.x, mat.cols-1)), max(0, min(p.y, mat.rows-1)));
}

template<typename T>
T sample(cv::Mat m, cv::Vec2f p) {
	using namespace cv;
	using namespace std;
	T r{ 0 };
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 2; i++) {
			Point p1(floor(j + p[0] - 0.5f), floor(i + p[1] - 0.5f));
			Vec2f d(1.f - abs(p1.x + 0.5f - p[0]), 1.f - abs(p1.y + 0.5f - p[1]));
			Point cp = clampToMat(p1, m);
			r += m.at<T>(cp) * d[0] * d[1];
		}
	}
	return r;
}