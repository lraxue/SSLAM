//
// Created by feixue on 17-7-19.
//

#ifndef SSLAM_IMAGEPROCESSOR_H
#define SSLAM_IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace SSLAM
{
    class ImageProcessor
    {
    public:
        // Normalized cross correlation between two image patches
        static float NCC(const cv::Mat& a, const cv::Mat& b);

        static int BruteForceMatch(const cv::Mat& mDesc1,
                                   const cv::Mat& mDesc2,
                                   std::vector<int>& vMatches);

        static int MatchWithRansac(const std::vector<cv::Point2f>& vKeys1, const cv::Mat& mDesc1,
                                   const std::vector<cv::Point2f>& vKeys2, const cv::Mat& mDesc2,
                                   std::vector<int>& vMatches);


    };
}

#endif //SSLAM_IMAGEPROCESSOR_H
