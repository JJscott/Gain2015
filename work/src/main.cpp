
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
#include "terrainopt.hpp"
#include "terrain.hpp"
#include "opencv_util.hpp"


using namespace cv;
using namespace std;






void testApprearanceSpace() {

	//gain::terrain test_terrain = gain::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);

	//// Appearance-space
	//Mat reduced = gain::appearanceSpace<4>(test_terrain.heightmap);

	//Mat ind[4];
	//split(reduced, ind);

	//imwrite(util::stringf("output/appearance_0.png"), gain::heightmapToImage(ind[0]));
	//imwrite(util::stringf("output/appearance_1.png"), gain::heightmapToImage(ind[1]));
	//imwrite(util::stringf("output/appearance_2.png"), gain::heightmapToImage(ind[2]));
	//imwrite(util::stringf("output/appearance_3.png"), gain::heightmapToImage(ind[3]));
}





void testK2Patchmatch() {
	gain::terrain test_terrain = gain::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);
	//// K=2 Patchmatch
	//Mat nnf = k2Patchmatch(testimage, testimage, 5, 4);

	//// convert offset coordinates to color image
	//Mat nnf_img_k1(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
	//Mat nnf_img_k2(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
	//for (int r = 0; r < nnf.rows; r++) {
	//	auto in_row = nnf.ptr<Vec<Vec2f, 2>>(r);
	//	auto out_row_k1 = nnf_img_k1.ptr<Vec3b>(r);
	//	auto out_row_k2 = nnf_img_k2.ptr<Vec3b>(r);
	//	for (int c = 0; c < nnf.cols; c++) {
	//		out_row_k1[c][2] = int((255.0 * in_row[c][0][0]) / testimage.cols); // cols -> r
	//		out_row_k1[c][1] = int((255.0 * in_row[c][0][1]) / testimage.rows); // rows -> g

	//		out_row_k2[c][2] = int((255.0 * in_row[c][1][0]) / testimage.cols); // cols -> r
	//		out_row_k2[c][1] = int((255.0 * in_row[c][1][1]) / testimage.rows); // rows -> g
	//	}
	//}

	//imwrite("output/NNF_k1.png", nnf_img_k1);
	//imwrite("output/NNF_k2.png", nnf_img_k2);
}




void testSynthesis() {

	// input terrain
	gain::terrain test_terrain = gain::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);
	
	// parameters
	gain::synthesis_params params;
	params.randomInit = true;

	Mat synth = gain::synthesizeTerrain(test_terrain.heightmap, params);
	//Mat synth = synthesizeTerrain(test_terrain.heightmap, Size(512, 512), 5, 7);
	imwrite("output/syth.png", gain::heightmapToImage(synth));
}






// main program
// 
int main( int argc, char** argv ) {

	//testApprearanceSpace();
	//testK2Patchmatch();
	testSynthesis();

}