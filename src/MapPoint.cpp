//
// Created by feixue on 17-6-12.
//

#include <MapPoint.h>
#include <include/ORBmatcher.h>
#include <glog/logging.h>

namespace SSLAM
{
    unsigned long MapPoint::mnNext = 0;
    MapPoint::MapPoint(Map* pMap, const Frame &frame, const int &idx):
            mpMap(pMap), mnLastFrameSeen(0),
            mnTrackLocalMapForFrame(0), mbBad(false),
            mnFuseMapPointForKF(0), nObs(0),
            mnFounders(0), mnAge(0), mTrackedAbility(0.f), mType(mTemporalPoint)
    {
        mnId = mnNext++;
//        mFeatureFlow[frame.mnId] = idx;

        // AddFounder(frame.mnId, idx); // Add the first founder

        mnReferenceFrame = frame.mnId;  // First appear
    }

    MapPoint::~MapPoint() {}

    void MapPoint::SetPos(const cv::Mat &pos)
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        mPos = pos.clone();
        //pos.copyTo(mPos);
    }

    cv::Mat MapPoint::GetPos()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mPos.clone();
    }

    cv::Mat MapPoint::GetDescriptor()
    {
        return mDescriptor.clone();
    }

    /**
     *
     * @param frameID this MapPoint is found by Frame with id of frameID
     * @param idx corresponded 2D feature point index
     */
    void MapPoint::AddFounder(const unsigned long &frameID, const int &idx)
    {
        mFeatureFlow[frameID] = idx;

        mnFounders += 1;  // For stereo cameras

        if (frameID == mnReferenceFrame)
            return;

        int nPassedFrames = frameID - mnReferenceFrame + 1;

        mnAge = nPassedFrames;
        mTrackedAbility = mnFounders / nPassedFrames;
    }

    float MapPoint::GetTrackAbility() const
    {
        return mTrackedAbility;
    }

    int MapPoint::GetAge() const
    {
        return mnAge;
    }


    std::map<unsigned long, int> MapPoint::GetAllFounders()
    {
        return mFeatureFlow;
    }

    void MapPoint::AddObservation(KeyFrame *pKF, const int &idx)
    {
        if (mObservations.count(pKF))
            return;

        mObservations[pKF] = idx;
        nObs += 1;

    }

    void MapPoint::EraseObservation(KeyFrame *pKF)
    {
        if (!pKF) return;
        if (!mObservations.count(pKF))
            return;

        nObs -= 2;
        mObservations.erase(pKF);



        // Observed by at least 2 stereo frames
        if (nObs <= 2)
            SetBad();
    }

    std::map<KeyFrame*, int> MapPoint::GetAllObservations() const
    {
        return mObservations;
    }

    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        if (!pKF) return -1;
        if (!mObservations.count(pKF)) return -1;

        return mObservations[pKF];
    }

    int MapPoint::Observations()
    {
        return nObs;
    }


    /****************************** Epipolar Triangle *********************/

    void MapPoint::AddTriangle(EpipolarTriangle *pTriangle)
    {
        if (!pTriangle)
            return;

        // Find if pTriangle exists
        std::list<EpipolarTriangle*>::iterator it = std::find(mlpTriangles.begin(), mlpTriangles.end(), pTriangle);
        if (it == mlpTriangles.end())
            mlpTriangles.push_back(pTriangle);
        else
            return;

//        LOG(INFO) << "MapPoint: " << mnId << " with " << mlpTriangles.size() << " triangles.";
    }

    void MapPoint::EraseTriangle(EpipolarTriangle *pTriangle)
    {
        if (!pTriangle)
            return;

        mlpTriangles.remove(pTriangle);
    }

    std::vector<EpipolarTriangle*> MapPoint::GetAllTriangles() const
    {
        // Return all related Triangles along the time line
        return std::vector<EpipolarTriangle*>(mlpTriangles.begin(), mlpTriangles.end());
    }

    EpipolarTriangle* MapPoint::GetLastEpipolarTriangle()
    {
        if (mlpTriangles.empty())
            return static_cast<EpipolarTriangle*>(NULL);
        else
            return mlpTriangles.back();
    }

    EpipolarTriangle* MapPoint::GetFirstEpipolarTriangle()
    {
        if (mlpTriangles.empty())
            return static_cast<EpipolarTriangle*>(NULL);
        else
            return mlpTriangles.front();
    }

    int MapPoint::Triangles()
    {
        // Return number of EpipolarTriangles associated to this MapPoint
        return mlpTriangles.size();
    }


    cv::Mat MapPoint::GetNormal()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    void MapPoint::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        std::vector<cv::Mat> vDescriptors;

        std::map<KeyFrame *, int> observations;

        {
            observations = mObservations;
        }

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for (std::map<KeyFrame *, int>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            KeyFrame *pKF = mit->first;

                vDescriptors.push_back(pKF->mDescriptorsLeft.row(mit->second));
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for (size_t i = 0; i < N; i++) {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++) {
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++) {
            std::vector<int> vDists(Distances[i], Distances[i] + N);
            sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian) {
                BestMedian = median;
                BestIdx = i;
            }
        }

            mDescriptor = vDescriptors[BestIdx].clone();
    }

    void MapPoint::UpdateNormalAndDepth()
    {
        std::map<KeyFrame *, int> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;

        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mPos.clone();

        if (observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int n = 0;
        for (std::map<KeyFrame *, int>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            KeyFrame *pKF = mit->first;
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mPos - Owi;
            normal = normal + normali / cv::norm(normali);
            n++;
        }

        cv::Mat PC = Pos - pRefKF->GetCameraCenter();
        const float dist = cv::norm(PC);
        const int level = pRefKF->mvKeysLeft[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels;

        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal / n;

//        LOG(INFO) << "KeyFrame id: " << mNormalVector;
    }

    float MapPoint::GetMinDistanceInvariance()
    {
        return 0.8 * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance()
    {
        return 1.2 * mfMaxDistance;
    }

    int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF) {
        float ratio;
        {
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pKF->mnScaleLevels)
            nScale = pKF->mnScaleLevels - 1;

        return nScale;
    }

    int MapPoint::PredictScale(const float &currentDist, Frame *pF) {
        float ratio;
        {
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }

    void MapPoint::SetBad()
    {
        mbBad = true;
    }

    bool MapPoint::IsBad()
    {
        return mbBad;
    }

    bool MapPoint::IsInKeyFrame(KeyFrame *pKF) const
    {
        return (mObservations.count(pKF));
    }

    void MapPoint::Replace(MapPoint *pMP)
    {
        if (pMP->mnId == mnId)   // The same MapPoint
            return;

        int nvisible, nfound;
        std::map<KeyFrame*, int> obs;
        obs = mObservations;
        mObservations.clear();
        mbBad = true;

        mpReplaced = pMP;

        for (auto mit : obs)
        {
            KeyFrame* pKF = mit.first;
            if (!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapPoint(mit.second, pMP);
                pMP->AddObservation(pKF, mit.second);
            }
            else
            {
                pKF->EraseMapPointByIndex(mit.second);
            }
        }

        pMP->ComputeDistinctiveDescriptors();
        mpMap->EraseMapPoint(this);
    }



}

