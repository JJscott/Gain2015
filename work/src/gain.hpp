#pragma once

// opencv
#include <opencv2/core.hpp>

namespace gain {
	cv::Mat synthesizeTerrain(cv::Mat example, cv::Size synth_size, int example_levels, int synth_levels);
	void test(cv::Mat testimage);
}