//
// Created by feixue on 17-7-13.
//

#ifndef SSLAM_FEATUREDETECTOR_H
#define SSLAM_FEATUREDETECTOR_H

#include <opencv2/opencv.hpp>


namespace SSLAM
{
    class FeatureDetector
    {
    public:
        static void DetectHarrisCorner(const cv::Mat& img, std::vector<cv::Point2f>& vHarrisCorners,
                                       const int& blockSize = 2, const int& apeartureSize = 3, const double& k = 0.04, const float& th = -1.f);

        static void DetectHarrisCorners(const cv::Mat& img, std::vector<cv::KeyPoint>& vHarrisCorners, const int& maxFeatures);

    protected:
        void SelectGoodFeatures(cv::InputArray _image, cv::OutputArray _corners, int maxCorners, double qualityLevel, double minDistance, int blockSize, double harrisK);
    };
}

#endif //SSLAM_FEATUREDETECTOR_H
