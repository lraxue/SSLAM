//
// Created by feixue on 17-6-28.
//

#include <ETFSLAM.h>
#include <glog/logging.h>

namespace SSLAM
{
    ETFSLAM::ETFSLAM(const std::string &strSettingFile)
    {
        GlobalParameters::LoadParameters(strSettingFile);

        mpMap = new Map();

        mpFrameDrawer = new FrameDrawer(mpMap);
        mpMapDrawer = new MapDrawer(mpMap);

        mpTracker = new Tracker(mpFrameDrawer, mpMapDrawer, mpMap);

        mpViewer = new Viewer(mpFrameDrawer, mpMapDrawer, mpTracker);

        ptrViewerThread = new thread(&Viewer::Run, mpViewer);
    }

    ETFSLAM::~ETFSLAM()
    {
        if (mpViewer)
            delete mpViewer;
        if (mpMapDrawer)
            delete mpMapDrawer;
        if (mpFrameDrawer)
            delete mpFrameDrawer;

        if (mpTracker)
            delete mpTracker;
        if (mpMap)
            delete mpMap;

        if (ptrViewerThread)
            delete ptrViewerThread;
    }

    void ETFSLAM::ProcessStereoImage(const cv::Mat &imLeft, const cv::Mat &imRight)
    {
        mpTracker->GrabStereo(imLeft, imRight);
    }
}
