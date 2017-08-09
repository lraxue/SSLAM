//
// Created by feixue on 17-6-14.
//

#ifndef SSLAM_OPTIMIZER_H
#define SSLAM_OPTIMIZER_H

#include <Frame.h>
#include <MapPoint.h>

namespace SSLAM
{
    class Optimizer
    {
    public:
        int static OptimizePose(Frame& frame);

        int static PoseOptimization(Frame *pFrame);

        void static LocalBundleAdjustment(KeyFrame* pKF, Map* pMap);

        // Bundle adjustment with uncertainty

        int static PoseOptimizationWithUncertainty(Frame* pFrame);

        void static LocalBundleAdjustmentWithUncertainty(KeyFrame* pKF, Map* pMap);


        // Pose optimization on 3D points
        int static OptimisePoseOn3DPoints(Frame &frame);

        // Local bundle adjustment based on 3D points
        void static LocalBundleAdjustmentBasedOn3DPoints(KeyFrame* pKF, Map* pMap);

        // ICP solver
        int static ICP(const std::vector<cv::Point3f>& vPoints1, const std::vector<cv::Point3f>& vPoints2, cv::Mat& R, cv::Mat& t, std::vector<bool>& vInliers);
    };
}

#endif //SSLAM_OPTIMIZER_H
