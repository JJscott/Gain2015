
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



	//Mat m(100, 100, CV_8UC3, Vec3b(0, 0, 0));
	//Mat c(200, 200, CV_32FC2);
	//Mat f(200, 200, CV_8UC3);


	//for (int i = 0; i < 100; i++) {
	//	for (int j = 0; j < 50; j++) {
	//		m.at<Vec3b>(Vec2i(i, j)) = Vec3b(255, 255, 255);
	//	}
	//}

	//for (int i = 0; i < 200; i++) {
	//	for (int j = 0; j < 200; j++) {
	//		c.at<Vec2f>(Vec2i(i, j)) = Vec2f(i/2+0.5, j / 2 + 0.5);
	//	}
	//}
	//remap(m, f, c, Mat(), INTER_LINEAR, BORDER_REPLICATE);


	//namedWindow("m");
	//imshow("m", m);

	//namedWindow("f");
	//imshow("f", f);

	//waitKey(0);
}