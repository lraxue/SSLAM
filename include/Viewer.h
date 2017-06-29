//
// Created by feixue on 17-6-28.
//

#ifndef SSLAM_VIEWER_H
#define SSLAM_VIEWER_H

#include <FrameDrawer.h>
#include <MapDrawer.h>

#include <mutex>

namespace SSLAM
{
    class Viewer
    {
    public:
        Viewer(FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracker* pTracker);

        // Main thread
        void Run();

        void RequestFinish();

        void RequestStop();

        bool isFinish();

        bool isStopped();

        void Release();

    private:
        FrameDrawer* mpFrameDrawer;
        MapDrawer* mpMapDrawer;

        Tracker* mpTracker;


        // 1/fps
        double mT;
        float mImageWidth, mImageHeight;

        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;


        bool Stop();
        bool CheckFinish();
        void SetFinish();
        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;

        bool mbStopped;
        bool mbStopRequested;
        std::mutex mMutexStop;
    };
}

#endif //SSLAM_VIEWER_H
