//
// Created by feixue on 17-6-28.
//

#ifndef SSLAM_ETFSLAM_H
#define SSLAM_ETFSLAM_H

#include <Tracker.h>
#include <Frame.h>
#include <Viewer.h>
#include <Map.h>
#include <LocalMapper.h>

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

        void Shutdown();

        void SaveTrajectoryKITTI(const std::string& strTrajectoryFile);

    public:
        /// Debugging functions
        void SaveAngleCorrespondedToOneMapPoint(const std::string& strAngleFile);

        /// Debugging functions

        void SaveObservationsInfo(const std::string& strObservationFile);

    public:
        Tracker* mpTracker;

        FrameDrawer* mpFrameDrawer;
        MapDrawer* mpMapDrawer;

        Viewer* mpViewer;
        LocalMapper* mpLocalMapper;

        Map* mpMap;


        thread* ptrViewerThread;


    };
}


#endif //SSLAM_ETFSLAM_H
