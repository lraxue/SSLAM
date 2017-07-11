//
// Created By FeiXue
//
// FeiXue@pku.edu.cn
//

#ifndef STEREOSLAM_FRAME_H
#define STEREOSLAM_FRAME_H

#include <ORBextractor.h>
#include <MapPoint.h>
#include <KeyFrame.h>
#include <EpipolarTriangle.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>


namespace SSLAM
{
    class MapPoint;
    class KeyFrame;
	class EpipolarTriangle;

	class Frame
	{
	public:
		Frame();
		// Copy constructor
		Frame(const Frame& frame);
		// Constructor
		Frame(const cv::Mat& imLeft, const cv::Mat& imRight);

		~Frame();

	public:
		void SetPose(const cv::Mat& Pos);

		cv::Mat GetPose() const;

        cv::Mat GetRotationInverse() const;

        cv::Mat GetCameraCenter() const;

        /** Tracking functions */
        // Search candidates in grid
        std::vector<int> SearchFeaturesInGrid(const float& cX, const float& cY, const float& radius, const int minLevel=-1, const int maxLevel=-1) const ;

		bool IsInFrustum(MapPoint* pMP, float viewingCosLimit);

        // Un-project 2D points to 3D space based on the pose of Frame
		cv::Mat UnprojectStereo(const int& idx);

		EpipolarTriangle* GenerateEpipolarTriangle(const int& idx) const;

        // Project corresponded point to 2D image
        cv::Point2f Project3DPointOnLeftImage(const int& idx) const;

		cv::Point2f Project3DPointOnRightImage(const int& idx) const;

        // Compute reprojection error
        float ComputeReprojectionError();

        void SaveDepthMap();

	protected:
		void UpdatePoseMatrix();

		// Extract features from left and right image
		void ExtractFeatures(const cv::Mat& im, const int& tag);

        // Assign features to grid for fast and robust matching
		void AssignFeaturesToGrid();

        // Is feature in grid
        bool PoseInGrid(const float& u, const float& v, int& x, int& y);

		// Compute stereo matches between left and right images
		void ComputeStereoMatches();

        // Record matched points, just for debugging
        void Record();

        void GenerateDisparityMap();

	public:
		// Frame ID
		unsigned long mnId;
		static unsigned long mnNext;

        KeyFrame* mpReferenceKF;

		// KeyPoints
		int N;
		std::vector<cv::KeyPoint> mvKeysLeft;
		std::vector<cv::KeyPoint> mvKeysRight;
		std::vector<cv::KeyPoint> mvKeysRightWithSubPixel;

		cv::Mat mDescriptorsLeft;
		cv::Mat mDescriptorsRight;

		std::vector<int> mvMatches;  // index of keypoints in right image, -1 default.
		std::vector<float> mvuRight;
		std::vector<float> mvDepth;
		std::vector<bool> mvbOutliers;

		// Corresponded MapPoints
        std::vector<MapPoint*> mvpMapPoints;

		// Corresponded EpipolarTriangles
		std::vector<EpipolarTriangle*> mvpTriangles;

		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvLogScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;

		// Calibration parameters 
		static float fx;
		static float fy;
		static float invfx;
		static float invfy;
		static float cx;
		static float cy;
		static float mb;
		static float mbf;
		static bool mbInitialization;

        // Grid size
        static int mnGridRows;
        static int mnGridCols;

		// Image size
		static int mnImgWidth;
		static int mnImgHeight;

        // Camera pose
        cv::Mat mTcw;
        cv::Mat mRcw;
        cv::Mat mRwc;
        cv::Mat mtcw;
        cv::Mat mOw; // == mtwc

        // Raw input image
        cv::Mat mLeft;
        cv::Mat mRight;
        cv::Mat mRGBLeft;
        cv::Mat mRGBRight;

        cv::Mat mDisp;

	protected:
		/// Feature parameters
		// Feature extractor
		std::shared_ptr<ORBextractor> mpORBextractorLeft;
		std::shared_ptr<ORBextractor> mpORBextractorRight;

        // Features in grid
        std::vector<std::vector<std::vector<int> > >mvFeaturesInGrid;

	};
}

#endif