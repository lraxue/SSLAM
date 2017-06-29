//
// Created by feixue on 17-6-12.
//

#include <Map.h>

namespace SSLAM
{
    Map::Map() {}
    Map::~Map() {}

    //***********************************Frame***********************************//
    void Map::AddFrame(const Frame &frame)
    {
        mvFrames.push_back(frame);
    }

    Frame Map::GetFrame(const unsigned long &idx)
    {
        return mvFrames[idx];
    }

    //***********************************MapPoint***********************************//
    void Map::AddMapPoint(MapPoint *pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    std::vector<MapPoint*> Map::GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
    }

    void Map::EraseMapPoint(MapPoint *pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        // NULL pointer or not exist, do nothing
        if (pMP == NULL || mspMapPoints.count(pMP) == 0)
            return;

        mspMapPoints.erase(pMP);
    }

    //***********************************Map***********************************//
    void Map::AddKeyFrame(KeyFrame *pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
    }

    std::vector<KeyFrame*> Map::GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        if (mspKeyFrames.empty())
            return std::vector<KeyFrame*>();

        return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    void Map::EraseKeyFrame(KeyFrame *pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        if (!pKF) return;
        if (!mspKeyFrames.count(pKF)) return;

        mspKeyFrames.erase(pKF);
    }




}