
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





	//auto sampleVec2f = [](cv::Mat m, cv::Vec2f p) -> Vec3b {
	//	using namespace cv;
	//	using namespace std;
	//	p += Vec2f(.5);
	//	Vec3b r(0, 0, 0);
	//	for (int j = 0; j < 2; j++) {
	//		for (int i = 0; i < 2; i++) {
	//			Point p1(floor(j + p[0] - 0.5f), floor(i + p[1] - 0.5f));
	//			Vec2f d(1.f - abs(p1.x + 0.5f - p[0]), 1.f - abs(p1.y + 0.5f - p[1]));
	//			Point cp = clampToMat(p1, m);
	//			r += m.at<Vec3b>(cp) * d[0] * d[1];
	//		}
	//	}
	//	return r;
	//};



	//Mat m(100, 100, CV_8UC3, Vec3b(0, 0, 0));
	//Mat c(100, 100, CV_32FC2);
	//Mat f1(100, 100, CV_8UC3);
	//Mat f2(100, 100, CV_8UC3);


	//for (int i = 0; i < 100; i++) {
	//	for (int j = 0; j < 50; j++) {
	//		m.at<Vec3b>(Point(j, i)) = Vec3b(255, 255, 255);
	//	}
	//}

	//for (int i = 0; i < 100; i++) {
	//	for (int j = 0; j < 100; j++) {
	//		c.at<Vec2f>(Point(j, i)) = Vec2f(j/100.0 + 49, i);
	//		f2.at<Vec3b>(Point(j, i)) = sampleVec2f(m, Vec2f(j / 100.0 + 49, i));
	//	}
	//}

	//remap(m, f1, c, Mat(), INTER_LINEAR, BORDER_REPLICATE);

	//for (int j = 0; j < 100; j++) {
	//	cout << norm(f1.at<Vec3b>(0, j), f2.at<Vec3b>(0, j)) << " : " << f1.at<Vec3b>(0, j) << " : " << f2.at<Vec3b>(0, j) << endl;
	//}



	//namedWindow("m");
	//imshow("m", m);

	//namedWindow("f1");
	//imshow("f1", f1);

	//namedWindow("f2");
	//imshow("f2", f2);

	//waitKey(0);
}