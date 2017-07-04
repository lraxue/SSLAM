//
// Created by feixue on 17-6-12.
//

#ifndef SSLAM_MAP_H
#define SSLAM_MAP_H

#include <Frame.h>
#include <MapPoint.h>
#include <KeyFrame.h>
#include <map>
#include <mutex>


namespace SSLAM
{
    class Frame;
    class MapPoint;
    class KeyFrame;

    class Map
    {
    public:
        Map();
        ~Map();

        // Frame processor
        void AddFrame(const Frame& frame);
        Frame GetFrame(const unsigned long& idx);

        // MapPoint processor
        void AddMapPoint(MapPoint* pMP);
        std::vector<MapPoint*> GetAllMapPoints();
        void EraseMapPoint(MapPoint* pMP);

        int GetMapPointsInMap();

        // KeyFrame processor
        void AddKeyFrame(KeyFrame* pKF);
        std::vector<KeyFrame*> GetAllKeyFrames();
        void EraseKeyFrame(KeyFrame* pKF);

        int GetKeyFramesInMap();


    protected:
        // Frames
        std::vector<Frame> mvFrames;

        // MapPoints
        std::set<MapPoint*> mspMapPoints;

        // KeyFrames
        std::set<KeyFrame*> mspKeyFrames;

        std::mutex mMutexMap;
    };
}
#endif //SSLAM_MAP_H
