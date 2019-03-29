#pragma once

// std
#include <vector>

// opencv
#include <opencv2/core.hpp>

namespace gain {

	struct constraint_value {
		bool valid;
		float height;
		float distance;
	};

	struct point_constraint {
		cv::Vec2f position;
		float height;
		float radius;
		cv::Vec2f gradient{0, 0};

		constraint_value calculateValue(cv::Vec2f pos, float maxRadius);
	};

	struct curve_node {
		int index;
		float height;
		float leftGradient;
		float rightGradient;
		float leftRadius;
		float rightRadius;
	};

	struct curve_constraint {
		std::vector<cv::Vec2f> path;
		std::vector<curve_node> curveNodes;
		
		constraint_value calculateValue(cv::Vec2f pos, float maxRadius);
	};


	struct synthesis_params {
		cv::Size synthesisSize{ 512, 512 };
		int synthesisLevels = 7; // M | (M > L)
		int exampleLevels = 5;  // L
		int correctionIter = 2;
		float jitter = 0.4;

		// constraints
		float constraintScale = 0.5; // distance scale for t(d) = sd
		float constraintSlope = 0.5; // slope at x >= abs(t(d))
		std::vector<point_constraint> pointConstraints;
		std::vector<curve_constraint> curveConstraints;
		

		// experimental parameters
		bool randomInit = false; // initialize the start of Gain with random coordinates
		bool scaleAppSpace = false; // scale the non-mean values of the apperance space appropriately
		enum {
			ENFORCE_FLOW_NONE,
			ENFORCE_FLOW_HEIGHTOFFSET,
			ENFORCE_FLOW_CONSTRAINT
		};
		int flowAlgorithm = ENFORCE_FLOW_NONE;
	};

	cv::Mat synthesizeTerrain(cv::Mat example, synthesis_params params);
}