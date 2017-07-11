//
// Created by feixue on 17-6-13.
//

#ifndef SSLAM_TRACKER_H
#define SSLAM_TRACKER_H

#include <Frame.h>
#include <MapPoint.h>
#include <Map.h>
#include <KeyFrame.h>
#include <FrameDrawer.h>
#include <MapDrawer.h>
#include <LocalMapper.h>

namespace SSLAM
{
    class FrameDrawer;

    class Tracker
    {
        enum mTrackingState
        {
            NOT_INITIALIZED = -1,
            SUCCEED = 1,
            LOST = 0,
        };
    public:
        Tracker(FrameDrawer* pFrameDrawer, MapDrawer *pMapDrawer, LocalMapper* pLocalMapper, Map* pMap);
        ~Tracker();

    public:
        cv::Mat GrabStereo(const cv::Mat& imLeft, const cv::Mat& imRight);

    protected:
        // Main track function
        void Track();

        // Track based on motion prediction
        bool TrackBasedOnMotionPrediction();


        // Track local map
        void UpdateLastFrame();

        void UpdateLocalKeyFrames();
        void UpdateLocalMapPoints();
        void UpdateLocalMap();
        void SearchLocalMapPoints();
        bool TrackLocalMap();

        // Create new KeyFrame
        void CreateNewKeyFrame();

        // Stereo initialization
        bool StereoInitialization();

    public:
        // Processed frame
        Frame mCurrentFrame;
        Frame mLastFrame;

        // Tracking state
        mTrackingState mState;

    protected:
        // Global Map for storing Frame, KeyFrame, MapPoints
        Map* mpMap;

        FrameDrawer* mpFrameDrawer;
        MapDrawer* mpMapDrawer;
        LocalMapper* mpLocalMapper;

        KeyFrame* mpReferenceKF;

        // Velocity for motion prediction
        cv::Mat mVelocity;

        float mThDepth;

        // Local map
        std::vector<KeyFrame*> mvpLocalKeyFrames;
        std::vector<MapPoint*> mvpLocalMapPoints;
    };
}

#endif //SSLAM_TRACKER_H
