#pragma once

// std
#include <string>

// opencv
#include <opencv2/core.hpp>

namespace gain {

	struct terrain {
		cv::Mat heightmap; // raster elevation
		double spacing; // real distance between adjacent samples
		terrain() : heightmap{}, spacing{0} { }
		terrain(cv::Mat _heightmap, double _spacing) : heightmap{ _heightmap }, spacing{ _spacing } {}
	};

	terrain terrainReadImage(const std::string &filename, double minVal, double maxVal, double spacing);
	terrain terrainReadTIFF(const std::string &filename);
	terrain terrainReadASC(const std::string &filename);
	
	void terrainWriteASC(const std::string &filename, terrain ter);

	cv::Mat heightmapToImage(cv::Mat heightmap);
	cv::Mat heightmapToImage(cv::Mat heightmap, double minv, double maxv);

}