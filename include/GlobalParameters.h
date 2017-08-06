//
// Created By FeiXue from Peking University.
//
// FeiXue@pku.edu.cn
//
//


#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include <string>

#ifndef USE_TRIANGLE
#define USE_TRIANGLE
#endif

//#ifndef DEBUG_RECORD
//#define DEBUG_RECORD
//#endif

//#ifndef DEBUG_DRAW_GRID
//#define DEBUG_DRAW_GRID
//#endif

#ifndef DEBUG_DRAW_STEREOMATCHES
#define DEBUG_DRAW_STEREOMATCHES
#endif

//#ifndef DEBUG_DRAW_MODIFIED
//#define DEBUG_DRAW_MODIFIED
//#endif

#ifndef DEBUG_DRAW_UNCERTAINTY
#define DEBUG_DRAW_UNCERTAINTY
#endif




namespace SSLAM
{
	// This class serves as a data class
	class GlobalParameters
	{
	public:
        // Load parameters
        void static LoadParameters(const std::string& strSettingFile);

	public:
		// Calibration parameters
		static float fx;
		static float fy;
		static float cx;
		static float cy;
		static float mb;
		static float mbf;

		static float mThDepth;

		static float mThAngle;

		static float mThMatch;

		// Features parameters
		static int mnLevels;
		static int mnFeatures;
		static float mfScaleFactor;
		static float iniThFAST;
		static float minThFAST;

		// Frame rates
		static float fps;

		// Self defined parameters
		static int mnGridRows;
        static int mnGridCols;

		// For showing map
		static float mKeyFrameSize;
		static float mKeyFrameLineWidth;
		static float mGraphLineWidth;
		static float mPointSize;
		static float mCameraSize;
		static float mCameraLineWidth;

        static float mImageWidth;
        static float mImageHeight;

        static float mViewpointX;
        static float mViewpointY;
        static float mViewpointZ;
        static float mViewpointF;

	};
}

#endif