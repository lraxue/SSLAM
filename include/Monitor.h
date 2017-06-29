//
// Created by feixue on 17-6-9.
//

#ifndef STEREO_SLAM_MONITOR_H
#define STEREO_SLAM_MONITOR_H

#include <opencv2/opencv.hpp>

namespace SSLAM
{

#define CIRCLE_RADIUS       10
#define CIRCLE_THICKNESS    5
#define LINE_THICKNESS      5
    class Monitor
    {
    public:
        /**
         *
         * @param mLeft left image
         * @param mRight right image
         * @param vPointsLeft matched points in left image
         * @param vPointsRight matched points in right image
         * @param out image with matched points
         */
        static void DrawMatchesBetweenStereoFrame(const cv::Mat mLeft, const cv::Mat& mRight,
                                           const std::vector<cv::KeyPoint>& vPointsLeft,
                                           const std::vector<cv::KeyPoint>& vPointsRight,
                                           cv::Mat& out);

        /**
         *
         * @param mLeft
         * @param mRight
         * @param vFeatureFlowLeft trajectory of features in left image
         * @param vFeatureFlowRight trajectory of features in right image
         * @param out image with feature flow
         */
        static void RecordFeatureFlow(const cv::Mat& mLeft, const cv::Mat& mRight,
                               const std::vector<std::vector<cv::KeyPoint> >& vFeatureFlowLeft,
                               const std::vector<std::vector<cv::KeyPoint> >& vFeatureFlowRight,
                               cv::Mat& out);

        // Measure disparity
        static float ComputeDisparityError(const cv::Mat& mD, const std::vector<cv::Point2f>& vLocations,
                                    const std::vector<float>& vDisparities);

        static void DrawKeyPoints(const cv::Mat& img, const std::vector<cv::Point2f>& points,
                                  const cv::Scalar& scalar, cv::Mat& out);

        static void DrawKeyPointsOnStereoFrame(const cv::Mat& mLeft, const cv::Mat& mRight,
                                               const std::vector<cv::KeyPoint>& vKeysLeft,
                                               const std::vector<cv::KeyPoint>& vKeysRight, cv::Mat& out);


        static void DrawMatchesWithModifiedPosition(const cv::Mat& mLeft, const cv::Mat& mRight,
                                                    const std::vector<cv::KeyPoint>& vKeysLeft,
                                                    const std::vector<cv::KeyPoint>& vKeysRight,
                                                    const std::vector<cv::KeyPoint>& vKeysRightModified, cv::Mat& out);

        static void DrawGridOnImage(const cv::Mat& img, const int& gridRows, const int& gridCols,
                                    cv::Mat& out, const cv::Scalar scalar = cv::Scalar(0, 0, 255));

        static void DrawReprojectedPointsOnStereoFrame(const cv::Mat& mLeft, const cv::Mat& mRight,
                                                       const std::vector<cv::KeyPoint>& vKeysLeft,
                                                       const std::vector<cv::KeyPoint>& vKeysRight,
                                                       const std::vector<cv::Point2f>& vProjectedKeysLeft,
                                                       const std::vector<cv::Point2f>& vProjectedKeysRight, cv::Mat& out);


        static void DrawReprojectedPointsOnStereoFrameAfterOptimization(const cv::Mat& mLeft, const cv::Mat& mRight,
                                                                        const std::vector<cv::KeyPoint>& vKeysLeft,
                                                                        const std::vector<cv::KeyPoint>& vKeysRight,
                                                                        const std::vector<cv::Point2f>& vProjectedKeysLeft,
                                                                        const std::vector<cv::Point2f>& vProjectedKeysRight,
                                                                        const std::vector<bool>& vInliers,
                                                                        cv::Mat& out);

        static void DrawMatchesBetweenTwoStereoFrames(const cv::Mat& mLastLeft, const cv::Mat& mLastRight,
                                                      const cv::Mat& mCurrLeft, const cv::Mat& mCurrRight,
                                                      const std::vector<cv::KeyPoint>& vLastKeysLeft,
                                                      const std::vector<cv::KeyPoint>& vLastKeysRight,
                                                      const std::vector<cv::KeyPoint>& vCurrKeysLeft,
                                                      const std::vector<cv::KeyPoint>& vCurrKeysRight,
                                                      cv::Mat& out);
    };
}

#endif //STEREO_SLAM_MONITOR_H
