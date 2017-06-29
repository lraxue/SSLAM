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
    };
}

#endif //SSLAM_OPTIMIZER_H
