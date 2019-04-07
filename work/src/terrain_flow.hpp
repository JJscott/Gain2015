
#pragma once

// std
#include <set>
#include <map>
#include <queue>
#include <functional>

// project
#include "terrain.hpp"



namespace scott {


	inline cv::Mat dirToImage(const cv::Mat dir) {
		using namespace cv;

		assert(!dir.empty());
		assert(dir.type() == CV_32SC2);

		Mat img(dir.rows, dir.cols, CV_8UC3, Scalar(0,0,0));
		for (size_t i = 0; i < dir.rows; i++) {
			for (size_t j = 0; j < dir.cols; j++) {
				Point p = dir.at<Point>(i, j);
				float d = 2 * norm(p);
				if (d > 0) {
					float a = (p.x + 1) / d;
					float b = (p.y + 1) / d;
					img.at<Vec3b>(i, j) =
						Vec3b(0, 255 * (1 - a), 255 * a) +
						Vec3b(255 * (1 - b), 255 * b, 255 * b);
				}
			}
		}

		return img;
	}



	inline cv::Mat d8FlowDirection(const cv::Mat elevation) {
		using namespace cv;
		using namespace std;

		assert(!elevation.empty());
		assert(elevation.type() == CV_32FC1);

		Mat direction(elevation.rows, elevation.cols, CV_32SC2);
		Rect bound(Point(0, 0), elevation.size());

		const Point dir[] = {
			Point{ -1, -1 }, Point{ -1, 0 }, Point{ -1, 1 },
			Point{ 0, -1 },                 Point{ 0, 1 },
			Point{ 1, -1 }, Point{ 1, 0 }, Point{ 1, 1 }
		};

		//#pragma omp parallel for
		for (int j = 0; j < elevation.cols; ++j) {
			for (int i = 0; i < elevation.rows; ++i) {
				Point p(j, i);
				float minGradient = 0; // lowest gradient
				Point minDir(0, 0);
				for (int n = 0; n < 8; ++n) {
					Point pn = p + dir[n];
					if (bound.contains(pn)) {
						float d = (elevation.at<float>(pn) - elevation.at<float>(p)) / norm(dir[n]);
						if (d < minGradient) {
							minGradient = d;
							minDir = dir[n];
						}
					}
					else if (minGradient == 0) {
						minDir = dir[n];
					}
				}
				direction.at<Point>(p) = minDir;
			}
		}

		return direction;
	}



	inline cv::Mat d8FlowAccumulation(const cv::Mat direction) {
		using namespace cv;

		assert(!direction.empty());
		assert(direction.type() == CV_32SC2);

		Mat accumulation(direction.rows, direction.cols, CV_32FC1, Scalar(-1));
		Rect bound(Point(0, 0), direction.size());

		const Point dir[] = {
			Point{ -1, -1 }, Point{ -1, 0 }, Point{ -1, 1 },
			Point{ 0, -1 },                 Point{ 0, 1 },
			Point{ 1, -1 }, Point{ 1, 0 }, Point{ 1, 1 }
		};

		// recursive function that calcuates accumulation of neighbours then sums for a given p
		std::function<void(const Point &)> calculate_accumulation = [&](const Point &p) {
			accumulation.at<float>(p) = 0;
			for (int n = 0; n < 8; ++n) {
				Point pn = p + dir[n];
				if (bound.contains(pn) && pn + direction.at<Point>(pn) == p) {
					if (accumulation.at<float>(pn) < 0) {
						calculate_accumulation(pn);
					}
					accumulation.at<float>(p) += accumulation.at<float>(pn) + 1;
				}
			}
		};

		for (size_t i = 0; i < direction.rows; ++i) {
			for (size_t j = 0; j < direction.cols; ++j) {
				calculate_accumulation(Point(j, i));
			}
		}

		return accumulation;
	}


	inline cv::Mat flowFillTerrain(cv::Mat elevation) {
		using namespace std;
		using namespace cv;

		assert(!elevation.empty());
		assert(elevation.type() == CV_32FC1);

		// make copy
		elevation = elevation.clone();

		const Point dir[] = {
			Point{ -1, -1 }, Point{ -1, 0 }, Point{ -1, 1 },
			Point{ 0, -1 },                 Point{ 0, 1 },
			Point{ 1, -1 }, Point{ 1, 0 }, Point{ 1, 1 }
		};

		struct elevation_cell {
			cv::Point position;
			float elevation; // needed for priority queue
			elevation_cell(cv::Point p, float e) : position(p), elevation(e) { }
		};

		// lower elevations had higher priority
		auto less = [](const elevation_cell &lhs, const elevation_cell &rhs) -> bool {
			return rhs.elevation < lhs.elevation;
		};

		priority_queue<elevation_cell, vector<elevation_cell>, decltype(less)> open(less);
		deque<Point> pit;
		Mat closed(elevation.size(), CV_8UC1);
		closed.setTo(Scalar(0));
		Rect bound(Point(0, 0), elevation.size());


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


	inline cv::Mat flowCarveTerrain(cv::Mat elevation) {
		using namespace cv;
		using namespace std;

		assert(!elevation.empty());
		assert(elevation.type() == CV_32FC1);

		// make copy
		elevation = elevation.clone();

		const Point dir[] = {
			Point{ -1, -1 }, Point{ -1, 0 }, Point{ -1, 1 },
			Point{ 0, -1 },                 Point{ 0, 1 },
			Point{ 1, -1 }, Point{ 1, 0 }, Point{ 1, 1 }
		};

		struct elevation_cell {
			cv::Point position;
			float elevation; // needed for priority queue
			elevation_cell(cv::Point p, float e) : position(p), elevation(e) { }
		};

		// lower elevations have higher priority
		auto less = [](const elevation_cell &lhs, const elevation_cell &rhs) -> bool {
			return rhs.elevation < lhs.elevation;
		};

		priority_queue<elevation_cell, vector<elevation_cell>, decltype(less)> open(less);
		Mat closed(elevation.size(), CV_8UC1);
		closed.setTo(Scalar(0));
		Mat parent(elevation.size(), CV_32SC2);
		parent.setTo(Scalar(0, 0));
		Rect bound(Point(0, 0), elevation.size());


		// push edge cells
		// left, right
		for (int i = 0; i < elevation.rows; ++i) {
			Point p1(i, 0);
			open.emplace(p1, elevation.at<float>(p1));
			closed.at<bool>(p1) = true;

			Point p2(i, elevation.cols - 1);
			open.emplace(p2, elevation.at<float>(p2));
			closed.at<bool>(p2) = true;
		}
		// top, bottom
		for (int j = 0; j < elevation.cols; ++j) {
			Point p1(0, j);
			open.emplace(p1, elevation.at<float>(p1));
			closed.at<bool>(p1) = true;

			Point p2(elevation.rows - 1, j);
			open.emplace(p2, elevation.at<float>(p2));
			closed.at<bool>(p2) = true;
		}


		while (!open.empty()) {
			Point p = open.top().position;
			open.pop();

			// for all neighbours
			for (int n = 0; n < 8; ++n) {
				Point pn = p + dir[n];

				// skip if closed
				if (!bound.contains(pn) || bool(closed.at<bool>(pn))) continue;
				closed.at<bool>(pn) = true;
				open.emplace(pn, elevation.at<float>(pn));
				parent.at<Point>(pn) = p;

				Point pit = pn;
				Point path = p;
				while (path != pit && elevation.at<float>(path) >= elevation.at<float>(pit)) {
					elevation.at<float>(path) = nexttoward(elevation.at<float>(pit), numeric_limits<float>::min());
					pit = path;
					path = parent.at<Point>(path);
				}
			}
		}

		return elevation;
	}





	//
	////calculate and return the flow fo the terrain
	////vec2 of flux (velocity)
	////vec1 of capacity
	//template <typename T>
	//cgra::arr<cgra::vec3> calculate_flow(const terrain::heightmap<T> &hm) {
	//	using namespace cgra;
	//
	//	arr<float> water_height(hm.elevation.size());
	//	arr<vec2> velocity(hm.elevation.size());
	//	arr<vec4> flux(hm.elevation.size()); // outflow from a given cell only
	//
	//	
	//	const ivec2 direction[] = {
	//		ivec2{  1,  0 }, // x
	//		ivec2{  0,  1 }, // y
	//		ivec2{ -1,  0 }, // x
	//		ivec2{  0, -1 }  // y
	//	};
	//
	//	//const float A = .04;
	//	const float A = 4;
	//	const float G = 10;
	//
	//	for (int q = 0; q < 5000; q++) {
	//
	//
	//		// increamenet water values (something small)
	//		for (size_t j = 0; j < water_height.size()[1]; ++j) {
	//			for (size_t i = 0; i < water_height.size()[0]; ++i) {
	//				//water_height.at(i, j) += 0.01;
	//				water_height.at(i, j) += 0.005;
	//			}
	//		}
	//
	//
	//		// update flux
	//		for (size_t j = 0; j < water_height.size()[1]; ++j) {
	//			for (size_t i = 0; i < water_height.size()[0]; ++i) {
	//				ivec2 p{ i, j };
	//
	//				// calculate/update flux
	//				float f_sum = 0;
	//				for (int n = 0; n < 4; ++n) {
	//					ivec2 pn = p + direction[n];
	//
	//					// difference in total height (elevation + capacity)
	//					float h = (water_height.inBounds(pn)) ? 
	//						hm.elevation.at(p) + water_height.at(p) - hm.elevation.at(pn) - water_height.at(pn) :
	//						water_height.at(p); // water height is used for h when pn is out of bounds
	//
	//					// flux is non negative 
	//					float temp_f = flux.at(p)[n] + A * G * h / hm.spacing;
	//					//float temp_f = A * G * h / hm.spacing; // convert to m^3/s flow (pipe area * gravity * height_diff / cell_distance)
	//					float f = max<float>(0, temp_f); // only allow positive outflow
	//					flux.at(p)[n] = f;
	//					f_sum += f; // sum flux
	//				}
	//
	//				if (f_sum > 0) {
	//					// adjust flux by the total sum
	//					T temp_k = water_height.at(p) * hm.spacing * hm.spacing / f_sum; // total volume / total flux
	//					T k = min<T>(1, temp_k); // max multiplier of 1 (so that flux doesn't increase)
	//					for (int n = 0; n < 4; ++n)
	//						flux.at(p)[n] = k * flux.at(p)[n];
	//				}
	//			}
	//		}
	//
	//
	//		// calcuate change in water height
	//		for (size_t j = 0; j < water_height.size()[1]; ++j) {
	//			for (size_t i = 0; i < water_height.size()[0]; ++i) {
	//				ivec2 p{ i, j };
	//				float dv = 0;
	//				for (int n = 0; n < 4; ++n) {
	//					
	//					dv -= flux.at(p)[n]; // p outflux
	//					
	//					ivec2 pn = p + direction[n];
	//					if (flux.inBounds(pn)) dv += flux.at(pn)[(n + 2) % 4]; // flux from pn->p
	//				}
	//
	//				// update water height (divide by surface to get height only)
	//				water_height.at(p) += dv / (hm.spacing * hm.spacing);
	//			}
	//		}
	//
	//
	//		// reduce water step maybe
	//		for (size_t j = 0; j < water_height.size()[1]; ++j) {
	//			for (size_t i = 0; i < water_height.size()[0]; ++i) {
	//				water_height.at(i, j) = max<float>(water_height.at(i, j) * 0.9, 0);
	//			}
	//		}
	//
	//	}
	//
	//
	//	// calculate velocities last
	//	// TODO move outside loop
	//	for (size_t j = 0; j < water_height.size()[1]; ++j) {
	//		for (size_t i = 0; i < water_height.size()[0]; ++i) {
	//			ivec2 p{ i, j };
	//			vec2 v;
	//
	//
	//			for (int n = 0; n < 4; ++n) {
	//				ivec2 pn = p + direction[n];
	//
	//				v += direction[n] * flux.at(p)[n];
	//				if (flux.inBounds(pn)) {
	//					v -= direction[n] * flux.at(pn)[(n + 2) % 4];
	//				}
	//			}
	//
	//			velocity.at(p) = v;
	//		}
	//	}
	//
	//
	//
	//	arr<float> vel_mag(hm.elevation.size());
	//	for (size_t j = 0; j < vel_mag.size()[1]; ++j) {
	//		for (size_t i = 0; i < vel_mag.size()[0]; ++i) {
	//			vel_mag.at(i, j) = length(velocity.at(i, j));
	//		}
	//	}
	//
	//
	//	gui::cache_upload(arr_to_heightmap(hm.elevation), "Elevation");
	//	gui::cache_upload(arr_to_better_heatmap(water_height), "Capacity");
	//	gui::cache_upload(arr_to_heatmap(vel_mag), "Magnitude of flux");
	//	gui::cache_upload(dir_to_image(velocity), "Velocity");
	//
	//
	//
	//
	//	return stack(stack(water_height), velocity);
	//}




	//
	////creates a flow feature map from a given elevation map
	//template <typename T>
	//cgra::arr<T> flow_feature(const cgra::arr<T> &elevation) {
	//
	//	// generate appearence space (with flow)
	//	arr<T> modified = priority_flood(elevatiaon);
	//	arr<ivec2> dir = d8_flow_direction(modified);
	//	arr<T> flow = d8_flow_accumulation<T>(dir);
	//
	//	for (int i = 0; i < flow.size()[0]; ++i) {
	//		for (int j = 0; j < flow.size()[1]; ++j) {
	//			T v = log(flow.at(i, j));
	//			if (v > 0) {
	//				flow.at(i, j) = log(flow.at(i, j)) * 1000;
	//			}
	//		}
	//	}
	//
	//	T flow_min = flow.min();
	//	T flow_max = flow.max();
	//	arr<T> flow_normalized = (flow - flow_min) / (flow_max - flow_min) / 3;
	//
	//	// TODO feature distance map
	//
	//	return flow_normalized;
	//}

}