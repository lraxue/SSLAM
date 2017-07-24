//
// Created by feixue on 17-7-10.
//

#include <LocalMapper.h>
#include <ORBmatcher.h>
#include <glog/logging.h>

namespace SSLAM
{
    LocalMapper::LocalMapper(Map *pMap): mpMap(pMap)
    {

    }

    LocalMapper::~LocalMapper()
    {
        // Nothing to do, embarrassed!
    }

    void LocalMapper::ProcessNewKeyFrame(KeyFrame *pKF)
    {
        mpCurrentKeyFrame = pKF;

        const std::vector<MapPoint*> vpMapPointsInCurrentKF = mpCurrentKeyFrame->GetMapPointMatches();
        for (int i = 0, iend = vpMapPointsInCurrentKF.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPointsInCurrentKF[i];
            if (pMP)
            {
                if (!pMP->IsBad())
                {
                    if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                    {
                        pMP->AddObservation(mpCurrentKeyFrame, i);
                        pMP->UpdateNormalAndDepth();
                        pMP->ComputeDistinctiveDescriptors();
                    }
                    else
                    {
                        // New created MapPoints
                        // TODO
                    }
                }
            }
        }

        // Update covisilbility graph
        mpCurrentKeyFrame->UpdateConnections();

        // First step
        FuseMapPoints();

        // Second local bundle adjustment
        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, mpMap);

        LOG(INFO) << "Process KeyFrame: " << mpCurrentKeyFrame->mnId << " finished.";
    }

    void LocalMapper::FuseMapPoints()
    {
        // Retrieve neighbor KeyFrames
        int nn = 10;
        const std::vector<KeyFrame*> vpNeighbors = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
        std::vector<KeyFrame*> vpTargetKFs;

        for (auto pKF1 : vpNeighbors)
        {
            if (pKF1->mnFuseMapPointsForKF == mpCurrentKeyFrame->mnId)
                continue;

            vpTargetKFs.push_back(pKF1);
            pKF1->mnFuseMapPointsForKF = mpCurrentKeyFrame->mnId;

            // Extend to second neighbor
            std::vector<KeyFrame*> vpSecondNeighbors = pKF1->GetBestCovisibilityKeyFrames(nn / 2);
            for (auto pKF2 : vpSecondNeighbors)
            {
                if (pKF2->mnFuseMapPointsForKF == mpCurrentKeyFrame->mnId ||
                        pKF2->mnId == mpCurrentKeyFrame->mnId)
                    continue;

                vpTargetKFs.push_back(pKF2);
                pKF2->mnFuseMapPointsForKF = mpCurrentKeyFrame->mnId;
            }
        }

        LOG(INFO) << "KeyFrames to fused: " << vpTargetKFs.size();

        // Search matches by projection from current KeyFrame to target KeyFrames
        ORBmatcher matcher;
        std::vector<MapPoint*> vpMapPointsInCurrentKF = mpCurrentKeyFrame->GetMapPointMatches();
        for (auto pKF : vpTargetKFs)
        {
            matcher.Fuse(pKF, vpMapPointsInCurrentKF);
        }


        // Search matches by projection form target KeyFrames to current KeyFrame
        std::vector<MapPoint*> vpCandidateMapPoints;
        vpCandidateMapPoints.reserve(vpTargetKFs.size() * vpMapPointsInCurrentKF.size());

        for (auto pKF : vpTargetKFs)
        {
            std::vector<MapPoint*> vpMapPointsInKF = pKF->GetMapPointMatches();

            for (auto pMP : vpMapPointsInKF)
            {
                if (!pMP)
                    continue;
                if (pMP->IsBad() || pMP->mnFuseMapPointForKF == mpCurrentKeyFrame->mnId)
                    continue;

                pMP->mnFuseMapPointForKF = mpCurrentKeyFrame->mnId;
                vpCandidateMapPoints.push_back(pMP);
            }
        }

        LOG(INFO) << "MapPoints to be fused: " << vpCandidateMapPoints.size();

        matcher.Fuse(mpCurrentKeyFrame, vpCandidateMapPoints);


        // Update MapPoints in current KeyFrame
        vpMapPointsInCurrentKF = mpCurrentKeyFrame->GetMapPointMatches();
        for (auto pMP : vpMapPointsInCurrentKF)
        {
            if (!pMP)
                continue;
            else if (pMP->IsBad())
                continue;

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();
        }

        // Update connections of current KeyFrame
        mpCurrentKeyFrame->UpdateConnections();

    }

}
