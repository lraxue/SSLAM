//
// Created by feixue on 17-6-28.
//

#ifndef SSLAM_FRAMEDRAWER_H
#define SSLAM_FRAMEDRAWER_H

#include <Map.h>
#include <Tracker.h>

#include <opencv2/opencv.hpp>
#include <mutex>

namespace SSLAM
{
    class Map;
    class Tracker;

    class FrameDrawer
    {
    public:
        FrameDrawer(Map* pMap);

        ~FrameDrawer();

        void Update(Tracker* pTracker);

        // Draw last processed frame
        cv::Mat DrawFrame();

    protected:
        void DrawTextInfo(cv::Mat& m, cv::Mat& imText);

        // Frame
        cv::Mat mIm;
        int N;
        std::vector<cv::KeyPoint> mvCurrentKeysLeft;
        std::vector<cv::KeyPoint> mvCurrentKeysRight;
        std::vector<cv::Point2f> mvCurrentProjectedKeysLeft;
        std::vector<cv::Point2f> mvCurrentProjectedKeysRight;

        std::vector<int> mvMatches;
        std::vector<bool> mvbMap;

        int mnTrackedMap, mnTrackedVO;

        Map* mpMap;

        unsigned long mnFrameId;

        std::mutex mMutex;

    };
}

#endif //SSLAM_FRAMEDRAWER_H
