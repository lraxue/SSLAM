//
// Created by feixue on 17-6-28.
//

#ifndef SSLAM_ETFSLAM_H
#define SSLAM_ETFSLAM_H

#include <Tracker.h>
#include <Frame.h>
#include <Viewer.h>
#include <Map.h>

#include <GlobalParameters.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <thread>

using namespace std;

namespace SSLAM
{
    class ETFSLAM
    {
    public:
        ETFSLAM(const std::string& strSettingFile);

        ~ETFSLAM();

    public:
        void ProcessStereoImage(const cv::Mat& imLeft, const cv::Mat& imRight);

        Tracker* mpTracker;

        FrameDrawer* mpFrameDrawer;
        MapDrawer* mpMapDrawer;

        Viewer* mpViewer;

        Map* mpMap;


        thread* ptrViewerThread;


    };
}


#endif //SSLAM_ETFSLAM_H
