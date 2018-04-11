
// std
#include <iostream>
#include <vector>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// project
#include "patchmatch.hpp"


using namespace cv;
using namespace std;


// Pixel & pixel, const int * position
Mat upsampleNNF(Mat nnf, Size size) {
	Mat up(size, CV_32SC2);

	for (int i = 0; i < up.rows; i++) {
		for (int j = 0; j < up.cols; j++) {
			//Point p(j / 2, i / 2);
			Point o(j % 2, i % 2);
			//up.at<Point>(i, j) = nnf.at<Point>(p) * 2 + o;
			up.at<Point>(i, j) = (nnf.at<Point>(i/2, j/2) * 2) + o;
		}
	}

	return up;
}



// main program
// 
int main( int argc, char** argv ) {

	// parameters
	const int patch_size = 5;
	const int patchmatch_iter = 4;
	const int pyramid_depth = 3;
	const int correction_iter = 4;


	// check we have exactly one additional argument
	// eg. res/vgc-logo.png
	if( argc != 3) {
		cout << "Usage: cgra352 sourceImage maskImage" << endl;
		abort();
	}

	// read the file
	Mat source_img, mask_img;
	source_img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	mask_img = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	// check for invalid input
	if(!source_img.data) {
		cout << "Could not open or find image for sourceImage argument" << std::endl;
		abort();
	}
	if (!mask_img.data) {
		cout << "Could not open or find image for targetImage argument" << std::endl;
		abort();
	}


	// make sure they're both the same size
	resize(mask_img, mask_img, source_img.size());
	//assert(source_img.size() == mask_img.size());


	// create the guess for the target
	Mat target_img = source_img.clone();
	for (int i = 0; i < target_img.rows; i++) {
		for (int j = 0; j < target_img.cols; j++) {
			if (mask_img.at<uchar>(i, j) > 128) {
				swap(target_img.at<Vec3b>(i, j), target_img.at<Vec3b>(i, j-270));
			}
		}
	}


	// construct pyramids
	vector<Mat> target_pyr, source_pyr;
	buildPyramid(target_img, target_pyr, pyramid_depth);
	buildPyramid(source_img, source_pyr, pyramid_depth);


	// iteration over pyramid
	Mat nnf;
	for (int k = pyramid_depth; k-->0;) {

		// correction
		for (int i = 0; i < correction_iter; i++) {
			nnf = patchmatch(source_pyr[k], target_pyr[k], patch_size, patchmatch_iter, nnf);
			target_pyr[k] = reconstruct(source_pyr[k], nnf, patch_size);
		}

		// upsample
		if (k > 0) {
			nnf = patchmatch(source_pyr[k], target_pyr[k], patch_size, patchmatch_iter);
			nnf = upsampleNNF(nnf, target_pyr[k - 1].size());
			target_pyr[k - 1] = reconstruct(source_pyr[k - 1], nnf, patch_size);
		}
	}



	// convert offset coordinates to color image
	Mat nnf_img(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
	for (int r = 0; r < nnf.rows; r++) {
		auto in_row = nnf.ptr<Point>(r);
		auto out_row = nnf_img.ptr<Vec3b>(r);
		for (int c = 0; c < nnf.cols; c++) {
			out_row[c][2] = int((255.0 * in_row[c].x) / source_img.cols); // cols -> r
			out_row[c][1] = int((255.0 * in_row[c].y) / source_img.rows); // rows -> g
			out_row[c][0] = 0;
		}
	}

	// reconstruct result
	Mat result = reconstruct(source_img, nnf, patch_size);

	imwrite("output/NNF.png", nnf_img);
	imwrite("output/Result.png", result);
}