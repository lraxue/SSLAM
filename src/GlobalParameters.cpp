//
// Created By FeiXue from Peking University.
//
// FeiXue@pku.edu.cn
//
//

#include "GlobalParameters.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace SSLAM
{
	// static parameters
	float GlobalParameters::fx = 0.0f;;
	float GlobalParameters::fy = 0.0f;
	float GlobalParameters::cx = 0.0f;
	float GlobalParameters::cy = 0.0f;
	float GlobalParameters::mb = 0.0f;
	float GlobalParameters::mbf = 0.0f;
	float GlobalParameters::fps = 0.0f;
	float GlobalParameters::iniThFAST = 0.0f;
	float GlobalParameters::minThFAST = 0.0f;
	float GlobalParameters::mfScaleFactor = 0.0f;
	int GlobalParameters::mnLevels = 0;
	int GlobalParameters::mnFeatures = 0;
	float GlobalParameters::mThDepth = 0.0f;

	// For Viewer
	float GlobalParameters::mKeyFrameSize = 0.0f;
	float GlobalParameters::mKeyFrameLineWidth = 0.0f;
	float GlobalParameters::mGraphLineWidth = 0.0f;
	float GlobalParameters::mPointSize = 0.0f;
	float GlobalParameters::mCameraSize = 0.0f;
	float GlobalParameters::mCameraLineWidth = 0.0f;

    float GlobalParameters::mImageWidth = 1280.0f;
    float GlobalParameters::mImageHeight = 720.0f;

    float GlobalParameters::mViewpointX = 0.0f;
    float GlobalParameters::mViewpointY = 0.0f;
    float GlobalParameters::mViewpointZ = 0.0f;
    float GlobalParameters::mViewpointF = 0.0f;

    // Self-defined parameters
	int GlobalParameters::mnGridCols = 64;
    int GlobalParameters::mnGridRows = 36;

	// Load parameters from file
	void GlobalParameters::LoadParameters(const std::string& strSettingFile)
	{
		cv::FileStorage fs = cv::FileStorage(strSettingFile.c_str(), cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			LOG(ERROR) << "Open setting file " << strSettingFile << " error.";
			exit(-1);
		}

		// Calibration parameters
		fx = fs["Camera.fx"];
		fy = fs["Camera.fy"];
		cx = fs["Camera.cx"];
		cy = fs["Camera.cy"];
		mbf = fs["Camera.bf"];
		mb = mbf / fx;

		fps = fs["Camera.fps"];

		mThDepth = fs["ThDepth"];

		mImageHeight = fs["Camera.height"];
		mImageWidth = fs["Camera.width"];

		// Feature parameters
		mnFeatures = fs["ORBextractor.nFeatures"];
		mnLevels = fs["ORBextractor.nLevels"];
		mfScaleFactor = fs["ORBextractor.scaleFactor"];
		iniThFAST = fs["ORBextractor.iniThFAST"];
		minThFAST = fs["ORBextractor.minThFAST"];

		mKeyFrameSize = fs["Viewer.KeyFrameSize"];
		mKeyFrameLineWidth = fs["Viewer.KeyFrameLineWidth"];
		mGraphLineWidth = fs["Viewer.GraphLineWidth"];
		mPointSize = fs["Viewer.PointSize"];
		mCameraSize = fs["Viewer.CameraSize"];
		mCameraLineWidth = fs["Viewer.CameraLineWidth"];

		mViewpointX = fs["Viewer.ViewpointX"];
		mViewpointY = fs["Viewer.ViewpointY"];
		mViewpointZ = fs["Viewer.ViewpointZ"];
		mViewpointF = fs["Viewer.ViewpointF"];



	}
}

