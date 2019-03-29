
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
	gain::terrain test_terrain = gain::terrainReadTIFF("work/res/mt_fuji_n035e138.tif");
	//gain::terrain test_terrain = gain::terrainReadTIFF("work/res/mount_jackson_n39_w107_3arc.tif");
	//gain::terrain test_terrain = gain::terrainReadTIFF("work/res/southern_alps_s045e169.tif");

	// parameters
	gain::synthesis_params params;
	//params.randomInit = true;
	params.synthesisSize = Size(512, 512);
	params.exampleLevels = 6;
	params.synthesisLevels = 8;


	double low, high;
	minMaxIdx(test_terrain.heightmap, &low, &high);

	//// random constraints test
	//for (int i = 0; i < 20; i++) {
	//	params.pointConstraints.push_back(gain::point_constraint{
	//		Vec2f(util::random<float>(0, 512), util::random<float>(0, 512)),
	//		util::random<float>(low + (high - low)/2, high), util::random<float>(16, 32)
	//	});
	//}


	params.curveConstraints.push_back(
		gain::curve_constraint{ {{287.985,138.957},{287.857,139.146},{287.607,139.522},{287.226,140.11},{286.737,140.891},{286.167,141.833},{285.539,142.9},{284.88,144.062},{284.211,145.297},{283.559,146.596},{282.948,147.963},{282.4,149.405},{281.925,150.927},{281.519,152.519},{281.161,154.161},{280.816,155.816},{280.445,157.445},{280.01,159.008},{279.485,160.481},{278.863,161.851},{278.154,163.126},{277.382,164.323},{276.583,165.465},{275.789,166.574},{275.028,167.666},{274.314,168.753},{273.645,169.838},{273.006,170.925},{272.369,172.016},{271.712,173.116},{271.018,174.231},{270.286,175.37},{269.529,176.538},{268.768,177.745},{268.026,178.996},{267.321,180.297},{266.664,181.648},{266.052,183.043},{265.469,184.464},{264.891,185.889},{264.292,187.29},{263.646,188.641},{262.935,189.923},{262.157,191.127},{261.318,192.253},{260.439,193.308},{259.542,194.301},{258.652,195.244},{257.791,196.154},{256.976,197.054},{256.216,197.975},{255.508,198.954},{254.846,200.019},{254.218,201.188},{253.612,202.463},{253.022,203.829},{252.444,205.261},{251.879,206.734},{251.325,208.224},{250.779,209.717},{250.234,211.2},{249.681,212.665},{249.109,214.101},{248.506,215.503},{247.866,216.865},{247.185,218.184},{246.464,219.464},{245.707,220.707},{244.919,221.919},{244.105,223.105},{243.271,224.271},{242.422,225.422},{241.566,226.566},{240.708,227.708},{239.852,228.853},{238.996,230},{238.137,231.146},{237.268,232.29},{236.383,233.43},{235.476,234.569},{234.538,235.709},{233.563,236.853},{232.544,237.998},{231.478,239.141},{230.376,240.279},{229.258,241.409},{228.158,242.533},{227.112,243.655},{226.161,244.784},{225.337,245.932},{224.664,247.112},{224.156,248.339},{223.821,249.622},{223.663,250.958},{223.685,252.326},{223.894,253.689},{224.293,254.995},{224.882,256.189},{225.644,257.225},{226.537,258.075},{227.493,258.731},{228.416,259.203},{229.199,259.514},{229.743,259.69},{230.033,259.772},},
{ {0, 813.063, 16.7701, 29.2648, 10, 10, },
{28, 920.625, 13.6859, 48.9824, 10, 10, },
{51, 1179.81, -7.46875, 22.8188, 10, 10, },
{91, 1574.87, 20.517, 61.4326, 10, 10, },
{104, 1922.92, -8.2207, -18.9674, 10, 10, },
} }
	);
	
	params.curveConstraints.push_back(
		gain::curve_constraint{ {{220.012,335.293},{220.325,335.162},{220.924,334.904},{221.811,334.504},{222.91,333.978},{224.132,333.347},{225.402,332.632},{226.662,331.848},{227.876,331.002},{229.023,330.098},{230.097,329.138},{231.107,328.126},{232.067,327.074},{233,325.999},{233.934,324.924},{234.895,323.87},{235.908,322.851},{236.994,321.875},{238.165,320.939},{239.431,320.031},{240.798,319.136},{242.263,318.239},{243.82,317.327},{245.449,316.395},{247.123,315.446},{248.807,314.483},{250.463,313.515},{252.061,312.551},{253.581,311.596},{255.016,310.655},{256.372,309.725},{257.662,308.796},{258.903,307.856},{260.108,306.893},{261.293,305.904},{262.47,304.899},{263.65,303.906},{264.844,302.964},{266.055,302.108},{267.283,301.362},{268.526,300.724},{269.781,300.169},{271.051,299.652},{272.344,299.126},{273.668,298.551},{275.025,297.904},{276.408,297.176},{277.794,296.373},{279.144,295.507},{280.406,294.59},{281.519,293.638},{282.427,292.668},{283.078,291.704},{283.439,290.776},{283.495,289.91},{283.249,289.126},{282.722,288.429},{281.945,287.807},{280.962,287.239},{279.819,286.698},{278.567,286.16},{277.253,285.604},{275.92,285.01},{274.602,284.366},{273.328,283.659},{272.114,282.881},{270.971,282.027},{269.901,281.098},{268.897,280.102},{267.946,279.053},{267.028,277.971},{266.122,276.878},{265.204,275.795},{264.258,274.742},{263.269,273.73},{262.231,272.77},{261.139,271.862},{259.999,271.006},{258.818,270.194},{257.611,269.418},{256.395,268.669},{255.185,267.942},{253.991,267.24},{252.811,266.574},{251.627,265.963},{250.403,265.427},{249.088,264.983},{247.632,264.638},{245.995,264.388},{244.162,264.22},{242.154,264.116},{240.015,264.057},{237.805,264.025},{235.58,264.01},{233.383,264.002},{231.235,263.996},{229.141,263.986},{227.1,263.963},{225.113,263.916},{223.188,263.824},{221.35,263.655},{219.631,263.371},{218.072,262.928},{216.713,262.287},{215.579,261.42},{214.675,260.324},{213.982,259.017},{213.46,257.54},{213.056,255.943},{212.72,254.278},{212.41,252.585},{212.098,250.889},{211.769,249.199},{211.414,247.517},{211.022,245.838},{210.578,244.159},{210.068,242.477},{209.484,240.786},{208.833,239.074},{208.143,237.331},{207.455,235.545},{206.815,233.711},{206.266,231.83},{205.833,229.907},{205.524,227.954},{205.335,225.982},{205.252,224.001},{205.267,222.022},{205.374,220.055},{205.573,218.114},{205.868,216.216},{206.259,214.379},{206.74,212.618},{207.294,210.941},{207.9,209.346},{208.53,207.815},{209.155,206.324},{209.753,204.845},{210.306,203.353},{210.809,201.831},{211.261,200.27},{211.663,198.667},{212.016,197.017},{212.322,195.323},{212.587,193.587},{212.823,191.823},{213.054,190.054},{213.311,188.311},{213.629,186.629},{214.039,185.039},{214.566,183.567},{215.224,182.226},{216.015,181.021},{216.936,179.953},{217.98,179.022},{219.131,178.23},{220.366,177.577},{221.639,177.062},{222.887,176.678},{224.023,176.409},{224.949,176.238},{225.576,176.143},{225.905,176.1},},
{ {0, 2986.88, -34.0125, -46.9707, 10, 10, },
{42, 2490.11, -41.8129, -6.63125, 10, 10, },
{55, 2876.88, -39.725, -38.2781, 10, 10, },
{87, 2445.86, 8.1125, -31.6406, 10, 10, },
{162, 2042.25, -24.4475, 4.95254, 10, 10, },
} }
	);



	Mat synth = gain::synthesizeTerrain(test_terrain.heightmap, params);
	imwrite("output/synthesis.png", gain::heightmapToImage(synth));
	gain::terrainWriteASC("output/synthesis.asc", gain::terrain(synth, test_terrain.spacing));
}



void testTerrainOutput() {

	// input terrain
	gain::terrain test = gain::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);
	gain::terrainWriteASC("output/test.asc", test);
}





static gain::terrain testTerrain;
static Mat color_img;
static bool curveActive = false;
static gain::curve_constraint constraint;


static void onMouse(int event, int x, int y, int, void*) {
	using namespace cv;
	using namespace std;

	const float radius = 10; // constraint radius

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
				//Vec2f left(-d[0], -d[1]);
				//Vec2f right(d[0], d[1]);
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
			cout << "{{";
			for (auto p : constraint.path) {
				cout << "{" << p[0] << "," << p[1] << "},";
			}
			cout << "}" << endl << "{";
			for (auto n : constraint.curveNodes) {
				cout << "{"
					<< n.index << ", "
					<< n.height << ", "
					<< n.leftGradient << ", "
					<< n.rightGradient << ", "
					<< n.leftRadius << ", "
					<< n.rightRadius << ", "
					<< "}," << endl;
			}
			cout << "}}" << endl;
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
			constraint.path.push_back(Vec2f(x, y));
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
	for (auto p : constraint.curveNodes) {
		circle(img, Point(constraint.path[p.index][0], constraint.path[p.index][1]), 3, Scalar(0, 255, 0), 1, LINE_AA);
	}

	imshow("img", img);
}

void constraintCreator() {

	//testTerrain = gain::terrainReadImage("work/res/mount_jackson.png", 0, 255, 1);
	testTerrain = gain::terrainReadTIFF("work/res/mt_fuji_n035e138.tif");
	//testTerrain = gain::terrainReadTIFF("work/res/mount_jackson_n39_w107_3arc.tif");
	//testTerrain = gain::terrainReadTIFF("work/res/southern_alps_s045e169.tif");

	Mat img = gain::heightmapToImage(testTerrain.heightmap);
	cvtColor(gain::heightmapToImage(testTerrain.heightmap), color_img, cv::COLOR_GRAY2BGR);

	namedWindow("img", WINDOW_AUTOSIZE);
	setMouseCallback("img", onMouse, 0);
	imshow("img", img);
	waitKey(0);
}


void testTerrainInputOutput() {


	gain::terrain test1 = gain::terrainReadTIFF("work/res/mt_fuji_n035e138.tif");
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
	testSynthesis();

	//testTerrainOutput();
	//testTerrainInputOutput();
}