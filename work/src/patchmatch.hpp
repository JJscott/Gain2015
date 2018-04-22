#pragma once

// opencv
#include <opencv2/core/core.hpp>

// returns a 2-channel matrix of the best offset
cv::Mat k2_patchmatch(cv::Mat source, cv::Mat target, int patch_size, float iterations = 1, cv::Mat est = {});

cv::Mat reconstruct(cv::Mat source, cv::Mat nnf, int patch_size);
cv::Mat nnfToImg(cv::Mat nnf, cv::Size s, bool absolute = true);