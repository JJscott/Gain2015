
#pragma once

// std
#include <cmath>
#include <queue>

// project
#include "terrain.hpp"



namespace scott {

	struct elevation_cell {
		cv::Point position;
		float elevation; // needed for priority queue
		elevation_cell(cv::Point p, float e) : position(p), elevation(e) { }
	};



	inline cv::Mat priority_flood_fill(cv::Mat elevation) {
		using namespace std;
		using namespace cv;

		assert(!elevation.empty());
		assert(elevation.type() == CV_32FC1);

		const Point dir[] = {
			Point{ -1, -1 }, Point{ -1, 0 }, Point{ -1, 1 },
			Point{ 0, -1 },                 Point{ 0, 1 },
			Point{ 1, -1 }, Point{ 1, 0 }, Point{ 1, 1 }
		};

		// lower elevations had higher priority
		auto less = [](const elevation_cell &lhs, const elevation_cell &rhs) -> bool {
			return rhs.elevation < lhs.elevation;
		};

		priority_queue<elevation_cell, vector<elevation_cell>, decltype(less)> open(less);
		deque<Point> pit;
		Mat closed(elevation.size(), CV_8UC1);
		closed.setTo(Scalar(0));
		Rect bound(Point(0,0), elevation.size());


		// push edge cells
		// left, right
		for (int i = 0; i < elevation.rows; ++i) {
			Point p1(i, 0);
			open.emplace(p1, elevation.at<float>(p1));
			closed.at<uchar>(p1) = true;

			Point p2(i, elevation.cols - 1);
			open.emplace(p2, elevation.at<float>(p2));
			closed.at<uchar>(p2) = true;
		}
		// top, bottom
		for (int j = 0; j < elevation.cols; ++j) {
			Point p1(0, j);
			open.emplace(p1, elevation.at<float>(p1));
			closed.at<uchar>(p1) = true;

			Point p2(elevation.rows - 1, j);
			open.emplace(p2, elevation.at<float>(p2));
			closed.at<uchar>(p2) = true;
		}


		while (!open.empty() || !pit.empty()) {
			Point p;
			// if there is no next pit cell
			// or if the next open cell == next pit cell
			if (pit.empty() || open.top().position == pit.front()) {
				// use the next open cell
				p = open.top().position;
				open.pop();
			}
			else {
				// use the next pit cell
				p = pit.front();
				pit.pop_front();
			}

			// for all neighbours
			for (int n = 0; n < 8; ++n) {
				Point pn = p + dir[n];

				// skip if closed, otherwise close
				if (!bound.contains(pn) || bool(closed.at<uchar>(pn))) continue;
				closed.at<uchar>(pn) = true;

				// compute a slightly higher elevation for the original cell
				float next_elevation = nextafter(elevation.at<float>(p), numeric_limits<float>::max());

				// if the neighbour is equal to or lower than next_elevation set it and add it to pit
				if (elevation.at<float>(pn) <= next_elevation) {
					elevation.at<float>(pn) = next_elevation;
					pit.emplace_back(pn);
				}

				// otherwise just add neighbour to open
				else {
					open.emplace(pn, elevation.at<float>(pn));
				}
			}
		}


		return elevation;
	}
}