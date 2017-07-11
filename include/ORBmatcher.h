//
// Created By FeiXue from Peking University.
//
// FeiXue@pku.edu.cn
//
//

#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <Frame.h>
#include <MapPoint.h>

#include <opencv2/opencv.hpp>

namespace SSLAM
{
	class ORBmatcher
	{
	public:

		ORBmatcher(float nnratio = 0.6, bool checkOri = true);

		// Computes the Hamming distance between two ORB descriptors
		static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

         /**
          * Search matched points from last Frame to current Frame based on projection
          * @param lastFrame Last Frame with global MapPoints
          * @param currentFrame Currently processed Frame
          * @param th Threshold for discard outliers
          * @return Number of searched points
          */
		int SearchByProjection(const Frame& lastFrame, Frame& currentFrame, const int th = 5);

		int SearchByProjection(Frame& F, const std::vector<MapPoint*>& vpMapPoints, const float& th);

		// Search circle matches
		int SearchCircleMatchesByProjection(const Frame &LastFrame, Frame &CurrentFrame, const float th);


		// Search based on Epipolar Triangles
		int SearchMatchesBasedOnEpipolarTriangles(const Frame& LastFrame, Frame& CurrentFrame, const float th);

		// Project MapPoints into KeyFrame and search for duplicated MapPoints
        int Fuse(KeyFrame* pKF, const std::vector<MapPoint*>& vpMapPoints, const float& th = 3.0);


	public:
		static const int TH_LOW;
		static const int TH_HIGH;
		static const int HISTO_LENGTH;

	protected:
		float RadiusByViewingCos(const float &viewCos);

		void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

		float mfNNratio;
		bool mbCheckOrientation;
	};
}
#endif