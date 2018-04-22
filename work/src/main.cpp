
// std
#include <iostream>
#include <vector>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// project
#include "patchmatch.hpp"
#include "gain.hpp"
#include "opencv_util.hpp"


using namespace cv;
using namespace std;






// main program
// 
int main( int argc, char** argv ) {

	// parameters
	const int patch_size = 7;
	const int patchmatch_iter = 4;


	// check we have exactly one additional argument
	// eg. work/res/mount_jackson.png
	if( argc != 2) {
		cout << "Usage: cgra352 sourceImage" << endl;
		abort();
	}

	// read the file
	Mat source_img;
	source_img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// check for invalid input
	if(!source_img.data) {
		cout << "Could not open or find image for sourceImage argument" << std::endl;
		abort();
	}

	
	Mat test_img(source_img.rows, source_img.cols, CV_32F);
	source_img.convertTo(test_img, CV_32F);
	gain::test(test_img);

}