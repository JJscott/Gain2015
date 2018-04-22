
// std
#include <iostream>
#include <string>

// project
#include "terrain.hpp"

// tiff
#include <tiffio.h>
#include <geotiffio.h>

// opencv
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;

// helper methods
namespace {



}

namespace gain {

	terrain terrainReadImage(const std::string &filename, double minVal, double maxVal, double spacing) {
	//	cgra::image<float, 1> img(filename);
	//	cgra::arr<T> t(img);

	//	T min_v = t.min();
	//	T max_v = t.max();

	//	cgra::arr<T> final_terrain = ((maxVal - minVal)*(t - min_v) / max_v) + minVal;

	//	return terrain<T>(final_terrain, spacing);
		return terrain();
	}



	// we make a huge number of assumuptions loading this data 
	// so we don't have to deal with the enormous number of cases
	// TODO list assumptions
	terrain terrainReadTIFF(const std::string &filename) {


		TIFF *tif;

		// Open the TIFF image for reading
		if ((tif = TIFFOpen(filename.c_str(), "r")) == NULL) {
			cerr << "File not found : " << filename << endl;
			throw runtime_error("File not found");
		}

		// Extract the 'sample' details for the TIFF
		uint16 bitsPerSample;        // normally 8 for grayscale image or 16 for heightmaps
		uint16 samplesPerPixel;      // normally 1 for grayscale image
		uint16 sampleFormat;         // should only be 1 for heightmaps
		TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
		TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat); // extension

		// determine matrix type (with bitshift magic)
		int mattype = 0;
		const int ls = 3; // number of bits to left shift by
		switch ((bitsPerSample << ls) | sampleFormat) {
		case (8 << ls) | SAMPLEFORMAT_INT: mattype = CV_8S; break;
		case (16 << ls) | SAMPLEFORMAT_INT: mattype = CV_16S; break;
		case (32 << ls) | SAMPLEFORMAT_INT: mattype = CV_32S; break;

		case (8 << ls) | SAMPLEFORMAT_UINT: mattype = CV_8U; break;
		case (16 << ls) | SAMPLEFORMAT_UINT: mattype = CV_16U; break;

		case (32 << ls) | SAMPLEFORMAT_IEEEFP: mattype = CV_32F; break;
		case (64 << ls) | SAMPLEFORMAT_IEEEFP: mattype = CV_64F; break;

		default: // image reading error
			cerr << "Invalid bitsPerSample=" << bitsPerSample << " and sampleFormat=" << sampleFormat << endl;
			throw runtime_error("Invalid bitsPerSample and sampleFormat combination.");
			break;
		}



		// get sizes and create image
		uint32 rows, cols;
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &rows);
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &cols);
		Mat heightmap(rows, cols, mattype);

		//allocate memory for reading tif image
		tdata_t raw_scanline = _TIFFmalloc(TIFFScanlineSize(tif));
		if (raw_scanline == NULL); // fatal error

		// reinterpret data as the right type
		int row_byte_size = (bitsPerSample * cols) / 8;
		for (uint32 row = 0; row < rows; row++) {
			//TIFFReadScanline(tif, raw_scanline, rows - row - 1);
			TIFFReadScanline(tif, raw_scanline, row);
			memcpy(heightmap.ptr(row), raw_scanline, row_byte_size);
		}

		_TIFFfree(raw_scanline); //free allocate memory




		//ModelPixelScaleTag     = 33550 (SoftDesk)
		//ModelTransformationTag = 33920 (Intergraph)
		//ModelTiepointTag       = 33922 (Intergraph)
		const int TIFFTAG_MODELPIXELSCALE = 33550;
		const int TIFFTAG_MODELTIEPOINT = 33922;


		uint64 count; // HACK formally unint16 but had stack corruption so relying on little eindian
		double *data;

		TIFFGetField(tif, TIFFTAG_MODELPIXELSCALE, &count, &data);
		Vec3d modelscale{ data[0], data[1], data[2] };

		// first 2 components (spacing converted from degrees to meters)
		// 1 degree = 110km (approx)
		Vec2d spacing = Vec2d{ modelscale[0], modelscale[1] } * 110000; 




		//// resize to a square if need be
		//if (spacing.x != spacing.y) {
		//	//TODO resize
		//	spacing.x = spacing.y = min(spacing.x, spacing.y);
		//}


		//if (modelscale.z > 0.0) {
		//	t *= modelscale.z;
		//}



		// TODO for later
		// adding a base resolution etc look up geotiff example for elevation maps
		//TIFFGetField(tif, TIFFTAG_MODELTIEPOINT, &count, &data);
		//vector<dvec3> model_ties;
		//for (uint16 i = 0; i < count / 3; ++i) {
		//	model_ties.emplace_back(data[3 * i + 0], data[3 * i + 1] , data[3 * i + 2]);
		//}


		//// geotiff stuff
		//GTIF *gtif = GTIFNew(tif);
		//GTIFKeyGet(gtif, something);
		//GTIFFree(gtif);


		// close TIFF
		TIFFClose(tif);


		Mat float_heightmap(heightmap.rows, heightmap.cols, CV_32F);
		heightmap.convertTo(float_heightmap, CV_32F);
		return terrain(float_heightmap, spacing[0]);
	}


	Mat heightmapToImage(Mat heightmap) {
		double minVal, maxVal;
		minMaxLoc(heightmap, &minVal, &maxVal);
		return heightmapToImage(heightmap, minVal, maxVal);
	}


	Mat heightmapToImage(Mat heightmap, double minv, double maxv) {
		Mat normalized = 255.0 * (heightmap - minv) / (maxv - minv);
		Mat img(normalized.rows, normalized.cols, CV_8U);
		normalized.convertTo(img, CV_8U);
		return img;
	}
}