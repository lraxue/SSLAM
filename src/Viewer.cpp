//
// Created by feixue on 17-6-28.
//

#include <Viewer.h>
#include <GlobalParameters.h>
#include <zconf.h>

namespace SSLAM
{
    Viewer::Viewer(FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracker *pTracker) :
            mpFrameDrawer(pFrameDrawer),
            mpMapDrawer(pMapDrawer), mpTracker(pTracker), mbFinishRequested(false), mbFinished(true),
            mbStopped(true), mbStopRequested(false)
    {
        float fps = GlobalParameters::fps;
        if (fps < 1)
            fps = 30;
        mT = 1e3 / fps;

        mImageWidth = GlobalParameters::mImageWidth;
        mImageHeight = GlobalParameters::mImageHeight;

        mViewpointX = GlobalParameters::mViewpointX;
        mViewpointY = GlobalParameters::mViewpointY;
        mViewpointZ = GlobalParameters::mViewpointZ;
        mViewpointF = GlobalParameters::mViewpointF;

    }

    void Viewer::Run()
    {
        mbFinished = false;
        mbStopped = false;

        pangolin::CreateWindowAndBind("SSLAM: Map Viewer", 1024, 768);

        // 3D mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1280, 720, mViewpointF, mViewpointF, 640, 360, 0.1, 1000),
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
        );

        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1280.0f / 720.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        cv::namedWindow("SSLAM: Current Frame");

        while (1)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            mpMapDrawer->GetOpenGLCameraMatrix(Twc);
            s_cam.Follow(Twc);

            d_cam.Activate(s_cam);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            mpMapDrawer->DrawCurrentCamera(Twc);
            mpMapDrawer->DrawKeyFrames(true);
            mpMapDrawer->DrawMapPoints();

            pangolin::FinishFrame();

            cv::Mat im = mpFrameDrawer->DrawFrame();
            cv::Mat resizeIm;
            cv::resize(im, resizeIm, cv::Size(1280, 360));
            cv::imshow("SSLAM: Current Frame", resizeIm);
            cv::waitKey(mT);


            if (Stop())
            {
                while(isStopped())
                {
                    usleep(3000);
                }
            }

            if (CheckFinish())
                break;
        }

        SetFinish();
    }

    void Viewer::RequestFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool Viewer::CheckFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void Viewer::SetFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool Viewer::isFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        return mbFinished;
    }

    void Viewer::RequestStop()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        if (!mbStopped)
            mbStopRequested = true;
    }


    bool Viewer::isStopped()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        return mbStopped;
    }

    bool Viewer::Stop()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        std::unique_lock<std::mutex> lock2(mMutexFinish);

        if(mbFinishRequested)
            return false;
        else if(mbStopRequested)
        {
            mbStopped = true;
            mbStopRequested = false;
            return true;
        }

        return false;
    }

    void Viewer::Release()
    {
        std::unique_lock<std::mutex> lock(mMutexStop);
        mbStopped = false;
    }
}
