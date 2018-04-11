
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



Mat appearance_space(Mat data) {
	assert(data.type() == CV_32FC1);

	const int numComponents = 4;

	// hardcoded normalized 5x5 gaussian kernal
	const float guassian_kernal[5][5] = {
		{ 0.00296902, 0.0133062, 0.0219382,  0.0133062, 0.00296902 },
		{ 0.0133062,  0.0596343, 0.0983203,  0.0596343, 0.0133062 },
		{ 0.0219382,  0.0983203, 0.16210312, 0.0983203, 0.0219382 },
		{ 0.0133062,  0.0596343, 0.0983203,  0.0596343, 0.0133062 },
		{ 0.00296902, 0.0133062, 0.0219382,  0.0133062, 0.00296902 }
	};

	// create the PCA set
	Mat neighbourhood_data(data.size(), CV_32FC(25));
	for (int i = 0; i < data.rows; i++) {
		for (int j = 0; j < data.cols; j++) {
			Point p(j, i);
			for (int ii = 0; ii < 5; ii++) {
				for (int jj = 0; jj < 5; jj++) {
					Point q(max(0, min(j+jj-2, data.cols - 1)), max(0, min(p.y+ii-2, data.rows - 1)));
					neighbourhood_data.at<Vec<Vec<float, 5>, 5>>(p)[ii][jj] = data.at<float>(q) * guassian_kernal[ii][jj];
				}
			}
			
		}
	}
	Mat pcaset = neighbourhood_data.reshape(1, data.cols * data.rows);

	// compute PCA
	PCA pca(pcaset, Mat(), PCA::DATA_AS_ROW, numComponents);


	// reduce, reshape, and return
	Mat reconstructed = pcaset * pca.eigenvectors.t();
	Mat reduced = reconstructed.reshape(numComponents, data.rows);
	return reduced;
}




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
	//source_img = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// check for invalid input
	if(!source_img.data) {
		cout << "Could not open or find image for sourceImage argument" << std::endl;
		abort();
	}






	//// convert offset coordinates to color image
	//Mat nnf = patchmatch(source_img, source_img, patch_size, patchmatch_iter);
	//Mat nnf_img(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
	//for (int r = 0; r < nnf.rows; r++) {
	//	auto in_row = nnf.ptr<Point>(r);
	//	auto out_row = nnf_img.ptr<Vec3b>(r);
	//	for (int c = 0; c < nnf.cols; c++) {
	//		out_row[c][2] = int((255.0 * in_row[c].x) / source_img.cols); // cols -> r
	//		out_row[c][1] = int((255.0 * in_row[c].y) / source_img.rows); // rows -> g
	//		out_row[c][0] = 0;
	//	}
	//}

	//// reconstruct result
	//Mat result = reconstruct(source_img, nnf, patch_size);

	//imwrite("output/NNF.png", nnf_img);
	//imwrite("output/Result.png", result);





	// Appearance-space

	Mat data(source_img.size(), CV_32FC1);
	source_img.convertTo(data, CV_32FC1);

	Mat reduced = appearance_space(data);
	cout << "channels : " << source_img.channels() << endl;
	cout << "data" << data(Range(0, 2), Range(0, 2)) << endl;
	cout << "reduced" << reduced(Range(0, 2), Range(0, 2)) << endl;


	Mat ind[4];
	split(reduced, ind);
	//for (int i = 0; i < 4; i++) normalize(ind[i], ind[i]);

	cout << "ind[0]" << ind[0](Range(0, 2), Range(0, 2)) << endl;

	cin.get();

	//imwrite("output/appearance_0.png", ind[0]);
	//imwrite("output/appearance_1.png", ind[1]);
	//imwrite("output/appearance_2.png", ind[2]);
	//imwrite("output/appearance_3.png", ind[3]);

	



	// K=2 Patchmatch


	//Mat nnf = k2_patchmatch(source_img, source_img, patch_size, patchmatch_iter);

	//// convert offset coordinates to color image
	//Mat nnf_img_k1(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
	//Mat nnf_img_k2(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
	//for (int r = 0; r < nnf.rows; r++) {
	//	auto in_row = nnf.ptr<Vec<Point, 2>>(r);
	//	auto out_row_k1 = nnf_img_k1.ptr<Vec3b>(r);
	//	auto out_row_k2 = nnf_img_k2.ptr<Vec3b>(r);
	//	for (int c = 0; c < nnf.cols; c++) {
	//		out_row_k1[c][2] = int((255.0 * in_row[c][0].x) / source_img.cols); // cols -> r
	//		out_row_k1[c][1] = int((255.0 * in_row[c][0].y) / source_img.rows); // rows -> g

	//		out_row_k2[c][2] = int((255.0 * in_row[c][1].x) / source_img.cols); // cols -> r
	//		out_row_k2[c][1] = int((255.0 * in_row[c][1].y) / source_img.rows); // rows -> g
	//	}
	//}

	//imwrite("output/NNF_k1.png", nnf_img_k1);
	//imwrite("output/NNF_k2.png", nnf_img_k2);
}