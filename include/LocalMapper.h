//
// Created by feixue on 17-6-16.
//

#ifndef SSLAM_LOCALMAPPER_H
#define SSLAM_LOCALMAPPER_H

#include <KeyFrame.h>
#include <MapPoint.h>
#include <Map.h>
#include <Optimizer.h>

namespace SSLAM
{
    class LocalMapper
    {
    public:
        LocalMapper(Map* pMap);
        ~LocalMapper();

    public:
        void ProcessNewKeyFrame(KeyFrame* pKF);

    };
}
#endif //SSLAM_LOCALMAPPER_H
