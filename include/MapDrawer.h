//
// Created by feixue on 17-6-28.
//

#ifndef SSLAM_MAPDRAWER_H
#define SSLAM_MAPDRAWER_H

#include <Map.h>
#include <MapPoint.h>
#include <KeyFrame.h>

#include <pangolin/pangolin.h>

#include <mutex>

namespace SSLAM
{
    class MapDrawer
    {
    public:
        MapDrawer(Map* pMap);

        ~MapDrawer();

        void DrawMapPoints();
        void DrawKeyFrames(const bool bDrawKF);
        void DrawCurrentCamera(pangolin::OpenGlMatrix& Twc);
        void SetCurrentCameraPose(const cv::Mat& Tcw);
        void SetReferenceKeyFrame(KeyFrame* pKF);
        void GetOpenGLCameraMatrix(pangolin::OpenGlMatrix& M);

    public:
        Map* mpMap;

    private:
        float mKeyFrameSize;
        float mKeyFrameLineWidth;
        float mGraphLineWidth;
        float mPointSize;
        float mCameraSize;
        float mCameraLineWidth;

        cv::Mat mCameraPose;

        std::mutex mMutexCamera;
    };
}

#endif //SSLAM_MAPDRAWER_H
