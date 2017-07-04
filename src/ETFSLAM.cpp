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
//        if (ptrViewerThread)
//            delete ptrViewerThread;
//        if (mpMapDrawer)
//            delete mpMapDrawer;
//        if (mpFrameDrawer)
//            delete mpFrameDrawer;
//
//        if (mpViewer)
//            delete mpViewer;
//
//        if (mpMap)
//            delete mpMap;
//
//        if (mpTracker)
//            delete mpTracker;


    }

    void ETFSLAM::ProcessStereoImage(const cv::Mat &imLeft, const cv::Mat &imRight)
    {
        mpTracker->GrabStereo(imLeft, imRight);
    }

    void ETFSLAM::Shutdown()
    {
        if (mpViewer)
        {
            mpViewer->RequestFinish();
            while (!mpViewer->isFinish())
            {
                usleep(5000);
            }
        }

        if (mpViewer)
            pangolin::BindToContext("SSLAM: Map Viewer");
    }
}
