//
// Created by feixue on 17-7-13.
//

#include <FeatureDetector.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using cv::Mat;

namespace SSLAM
{
    void FeatureDetector::DetectHarrisCorner(const cv::Mat &img, std::vector<cv::Point2f> &vHarrisCorners,
                                             const int &blockSize, const int& apeartureSize, const double& k, const float &th)
    {
        assert(img.channels() == 1);

//        Mat dst;
//        dst = cv::Mat::zeros(img.size(), CV_32FC1);
//
//        // Detect corners
//        cv::cornerHarris(img, dst, blockSize, apeartureSize, k, cv::BORDER_DEFAULT);
//        // Normalize
//        Mat dst_norm, dst_norm_scaled;
//        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, Mat());
//        cv::convertScaleAbs(dst_norm, dst_norm_scaled);

        // Extract corners

    }

    void FeatureDetector::DetectHarrisCorners(const cv::Mat &img, std::vector<cv::KeyPoint> &vHarrisCorners,
                                              const int &maxFeatures)
    {
        assert(img.channels() == 1);

        cv::GFTTDetector detector(maxFeatures);

        detector.detect(img, vHarrisCorners, Mat());
    }

    void FeatureDetector::SelectGoodFeatures(const cv::_InputArray &_image, const cv::_OutputArray &_corners,
                                             int maxCorners, double qualityLevel, double minDistance, int blockSize,
                                             double harrisK)
    {
        // TODO
    }
}

