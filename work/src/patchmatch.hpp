#pragma once

// opencv
#include <opencv2/core/core.hpp>


cv::Mat patchmatch(cv::Mat source, cv::Mat target, int patch_size, float iterations, cv::Mat est = {});

// returns a 2-channel matrix of the best offset
// final result is Vec<Vec2f, 2> where each Vec2f represents (col, row)!
// nnf is a mapping of each point in the target to the best match in the source
cv::Mat k2Patchmatch(cv::Mat source, cv::Mat target, int patch_size, float iterations = 1, cv::Mat est = {});

cv::Mat reconstruct(cv::Mat source, cv::Mat nnf, int patch_size);
cv::Mat nnfToImg(cv::Mat nnf, cv::Size s, bool absolute = true);