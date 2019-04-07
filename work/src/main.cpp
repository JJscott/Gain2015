
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
	//gain::terrain test_terrain = gain::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);
	//gain::terrain test_terrain = gain::terrainReadTIFF("work/res/mt_fuji_n035e138.tif");
	//gain::terrain test_terrain = gain::terrainReadTIFF("work/res/mount_jackson_n39_w107_3arc.tif");
	//gain::terrain test_terrain = gain::terrainReadTIFF("work/res/southern_alps_s045e169.tif");
	gain::terrain test_terrain = gain::terrainReadASC("output/source.asc");

	// parameters
	gain::synthesis_params params;
	//params.randomInit = true;
	params.synthesisSize = Size(1024, 1024);
	params.exampleLevels = 7;
	params.synthesisLevels = 9;


	double low, high;
	minMaxIdx(test_terrain.heightmap, &low, &high);

	//// random constraints test
	//for (int i = 0; i < 20; i++) {
	//	params.pointConstraints.push_back(gain::point_constraint{
	//		Vec2f(util::random<float>(0, 512), util::random<float>(0, 512)),
	//		util::random<float>(low + (high - low)/2, high), util::random<float>(16, 32)
	//	});
	//}


	Mat synth = gain::synthesizeTerrain(test_terrain.heightmap, params);
	imwrite("output/synthesis.png", gain::heightmapToImage(synth));
	gain::terrainWriteASC("output/synthesis.asc", gain::terrain(synth, test_terrain.spacing));
}







static gain::terrain testTerrain;
static Mat color_img;
static bool curveActive = false;
static gain::curve_constraint constraint;


static void onMouse(int event, int x, int y, int, void*) {
	using namespace cv;
	using namespace std;

	const float radius = 20; // constraint radius

	switch (event) {
	case EVENT_LBUTTONDOWN:
		if (!curveActive) {
			constraint = gain::curve_constraint();
		}
		curveActive = true;
		constraint.curveNodes.push_back({int(constraint.path.size())});
		constraint.path.push_back(Vec2f(x, y));
		break;
	case EVENT_RBUTTONDOWN:
		if (curveActive) {
			constraint.curveNodes.push_back({ int(constraint.path.size()) });
			constraint.path.push_back(Vec2f(x, y));

			// smooth
			for (int s = 0; s < 10; s++) {
				const float kern[] = { .006, .061, .242, .383, .242, .061, .006 };
				std::vector<cv::Vec2f> newPath;
				for (int i = 0; i < constraint.path.size(); i++) {
					Vec2f sum = 0;
					float weight = 0;
					for (int k = 0; k < 7; k++) {
						int c = i + k - 3;
						if (c >= 0 && c < constraint.path.size()) {
							sum += constraint.path[c] * kern[k];
							weight += kern[k];
						}
					}

					newPath.push_back(sum / weight);
				}
				constraint.path = newPath;
			}

			// calculate gradients
			for (auto &node : constraint.curveNodes) {
				Vec2f p1 = constraint.path[node.index];
				Vec2f p2, d;
				if (node.index + 1 < constraint.path.size()) {
					p2 = constraint.path[node.index + 1];
					d = p2 - p1;
				}
				else {
					p2 = constraint.path[node.index - 1];
					d = p1 - p2;
				}
				d /= norm(d);

				// sample gradients on either side
				Vec2f left(-d[1], d[0]);
				Vec2f right(d[1], -d[0]);
				float sampleRadius = radius / 2;

				Mat sampler(3, 1, CV_32FC2), sample;
				sampler.at<Vec2f>(0, 0) = p1;
				sampler.at<Vec2f>(1, 0) = left * sampleRadius + p1;
				sampler.at<Vec2f>(2, 0) = right * sampleRadius + p1;
				remap(testTerrain.heightmap, sample, sampler, Mat(), INTER_LINEAR, BORDER_REPLICATE);

				node.height = sample.at<float>(0, 0);
				node.leftGradient = (sample.at<float>(1, 0) - node.height) / sampleRadius;
				node.rightGradient = (sample.at<float>(2, 0) - node.height) / sampleRadius;
				node.leftRadius = radius;
				node.rightRadius = radius;
			}

			// print
			cout << "gain::curve_constraint{{";
			for (auto p : constraint.path) {
				cout << "{" << p[0] << "," << p[1] << "},";
			}
			cout << "},{";
			for (auto n : constraint.curveNodes) {
				cout << "{"
					<< n.index << ", "
					<< n.height << ", "
					<< n.leftGradient << ", "
					<< n.rightGradient << ", "
					<< n.leftRadius << ", "
					<< n.rightRadius << ", "
					<< "},";
			}
			cout << "}}" << endl;

			for (int i = 0; i < int(constraint.path.size()) - 1; i++) {
				line(color_img, Point(constraint.path[i][0], constraint.path[i][1]), Point(constraint.path[i + 1][0], constraint.path[i + 1][1]), Scalar(0, 255, 0), 1, LINE_AA);
			}
			for (auto node : constraint.curveNodes) {
				// center and profile
				Vec2f p = constraint.path[node.index], d;
				circle(color_img, Point(p[0], p[1]), 3, Scalar(0, 255, 0), 1, LINE_AA);
				if (node.index + 1 < constraint.path.size())
					d = constraint.path[node.index + 1] - p;
				else
					d = p - constraint.path[node.index - 1];
				d /= norm(d);
				Vec2f left = p + Vec2f(-d[1], d[0]) * node.leftRadius;
				Vec2f right = p + Vec2f(d[1], -d[0]) * node.rightRadius;

				line(color_img, Point(p[0], p[1]), Point(left[0], left[1]), Scalar(255, 0, 0), 1, LINE_AA);
				line(color_img, Point(p[0], p[1]), Point(right[0], right[1]), Scalar(255, 0, 0), 1, LINE_AA);
			}


		}
		else {
			cout << "sample (" << x << "," << y << ")" << endl;
			gain::constraint_value c = constraint.calculateValue(Vec2f(x, y), 32);
			cout << "distance : " << c.distance<< endl;
			cout << "height   : " << c.height << endl;
			cout << "valid    : " << c.valid << endl;
		}
		curveActive = false;
		break;
	case EVENT_MOUSEMOVE: 
		if (curveActive) {
			static int c = 0;
			if (c++ % 3 == 0) {
				constraint.path.push_back(Vec2f(x, y));
			}
		}
		break;
	default:
		return;
	};


	
	static Mat img;
	color_img.copyTo(img);
	

	for (int i = 0; i < int(constraint.path.size()) - 1; i++) {
		line(img, Point(constraint.path[i][0], constraint.path[i][1]), Point(constraint.path[i+1][0], constraint.path[i + 1][1]), Scalar(0, 0, 255), 1, LINE_AA);
	}
	for (auto node : constraint.curveNodes) {
		Vec2f p = constraint.path[node.index], d;
		circle(img, Point(p[0], p[1]), 3, Scalar(0, 255, 0), 1, LINE_AA);
	}

	imshow("img", img);
}

void constraintCreator() {

	//testTerrain = gain::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);
	//testTerrain = gain::terrainReadTIFF("work/res/mt_fuji_n035e138.tif");
	//testTerrain = gain::terrainReadTIFF("work/res/mount_jackson_n39_w107_3arc.tif");
	//testTerrain = gain::terrainReadTIFF("work/res/southern_alps_s045e169.tif");
	testTerrain = gain::terrainReadASC("output/target.asc");

	for (int i = 0; i < 10; i++)
		GaussianBlur(testTerrain.heightmap, testTerrain.heightmap, Size(31, 31), 1);


	Mat img = gain::heightmapToImage(testTerrain.heightmap);
	cvtColor(gain::heightmapToImage(testTerrain.heightmap), color_img, cv::COLOR_GRAY2BGR);

	namedWindow("img", WINDOW_AUTOSIZE);
	setMouseCallback("img", onMouse, 0);
	imshow("img", img);
	waitKey(0);
}





void downsampleCreator() {
	gain::terrain test = gain::terrainReadASC("output/jack_target.asc");
	//resize(test.heightmap, test.heightmap, Size(64, 64), 0, 0, INTER_CUBIC);
	//resize(test.heightmap, test.heightmap, Size(1000, 1000), 0, 0, INTER_CUBIC);
	for (int i = 0; i < 250; i++)
		GaussianBlur(test.heightmap, test.heightmap, Size(7, 7), 1);
	gain::terrainWriteASC("output/jack_target_scott.asc", test);
}









void testTerrainOutput() {

	// input terrain
	//gain::terrain test = gain::terrainReadTIFF("work/res/mount_jackson_n39_w107_1arc_v3.tif");

	gain::terrain test, source, target;

	test = gain::terrainReadTIFF("output/mount_jackson_n39_w107_1arc_v3.tif");
	source = gain::terrain(test.heightmap(Range(2600, 3600), Range(1000, 2000)), test.spacing);
	target = gain::terrain(test.heightmap(Range(1600, 2600), Range(1000, 2000)), test.spacing);
	gain::terrainWriteASC("output/jack_full.asc", test);
	gain::terrainWriteASC("output/jack_source.asc", source);
	gain::terrainWriteASC("output/jack_target.asc", target);



	test = gain::terrainReadTIFF("output/moldoveanu_peak_n45_e024_1arc_v3.tif");
	source = gain::terrain(test.heightmap(Range(600, 1600), Range(1600, 2600)), test.spacing);
	target = gain::terrain(test.heightmap(Range(600, 1600), Range(2600, 3600)), test.spacing);
	gain::terrainWriteASC("output/mold_full.asc", test);
	gain::terrainWriteASC("output/mold_source.asc", source);
	gain::terrainWriteASC("output/mold_target.asc", target);


	test = gain::terrainReadTIFF("output/mount_sumbing_s08_e109_110_1arc_v3.tif");
	source = gain::terrain(test.heightmap(Range(900, 1900), Range(3400, 4400)), test.spacing);
	target = gain::terrain(test.heightmap(Range(1300, 2300), Range(4700, 5700)), test.spacing);
	gain::terrainWriteASC("output/sumb_full.asc", test);
	gain::terrainWriteASC("output/sumb_source.asc", source);
	gain::terrainWriteASC("output/sumb_target.asc", target);


	test = gain::terrainReadTIFF("output/mount_woodroffe_s27_e131_1arc_v3.tif");
	source = gain::terrain(test.heightmap(Range(500, 1500), Range(1300, 2300)), test.spacing);
	target = gain::terrain(test.heightmap(Range(500, 1500), Range(2300, 3300)), test.spacing);
	gain::terrainWriteASC("output/wood_full.asc", test);
	gain::terrainWriteASC("output/wood_source.asc", source);
	gain::terrainWriteASC("output/wood_target.asc", target);


	test = gain::terrainReadTIFF("output/al-hadah_n16_e048_1arc_v3.tif");
	source = gain::terrain(test.heightmap(Range(2500, 3500), Range(0, 1000)), test.spacing);
	target = gain::terrain(test.heightmap(Range(2500, 3500), Range(1000, 2000)), test.spacing);
	gain::terrainWriteASC("output/hada_full.asc", test);
	gain::terrainWriteASC("output/hada_source.asc", source);
	gain::terrainWriteASC("output/hada_target.asc", target);
}



void testTerrainInputOutput() {


	gain::terrain test1 = gain::terrainReadTIFF("work/res/mount_jackson_n39_w107_1arc_v3.tif");
	gain::terrainWriteASC("output/testoutput.asc", test1);
	gain::terrain test2 = gain::terrainReadASC("output/testoutput.asc");

	float err = norm(test1.heightmap - test1.heightmap);
	cout << "test1 : (" << test1.heightmap.cols << ", " << test1.heightmap.rows << ") : " << test1.spacing << endl;
	cout << "test2 : (" << test2.heightmap.cols << ", " << test2.heightmap.rows << ") : " << test2.spacing << endl;
	cout << "err " << err << endl;

}









// main program
// 
int main( int argc, char** argv ) {

	//testApprearanceSpace();
	//testK2Patchmatch();

	//constraintCreator();
	//testSynthesis();

	downsampleCreator();
	//testTerrainOutput();
	//testTerrainInputOutput();
}