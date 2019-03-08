#pragma once

// opencv
#include <opencv2/core.hpp>

namespace gain {

	struct synthesis_params {
		cv::Size synthesisSize{ 512, 512 };
		int synthesisLevels = 7; // M | (M > L)
		int exampleLevels = 5;  // L
		int correctionIter = 2;
		float jitter = 0.4;

		// constraints
		

		// experimental parameters
		bool randomInit = false; // initialize the start of Gain with random coordinates
		bool scaleAppSpace = false; // scale the non-mean values of the apperance space appropriately
		enum {
			ENFORCE_FLOW_NONE
		};
		int flowAlgorithm = ENFORCE_FLOW_NONE;
	};

	cv::Mat synthesizeTerrain(cv::Mat example, synthesis_params params);
}