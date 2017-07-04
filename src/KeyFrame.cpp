//
// Created by feixue on 17-6-15.
//

#include <KeyFrame.h>
#include <GlobalParameters.h>
#include <glog/logging.h>

namespace SSLAM
{
    // Static parameters
    float KeyFrame::fx = 0.0f;
    float KeyFrame::fy = 0.0f;
    float KeyFrame::invfx = 0.0f;
    float KeyFrame::invfy = 0.0f;
    float KeyFrame::cx = 0.0f;
    float KeyFrame::cy = 0.0f;
    float KeyFrame::mb = 0.0f;
    float KeyFrame::mbf = 0.0f;
    bool KeyFrame::mbInitialization = false;

    unsigned long KeyFrame::mnNext = 0;
    int KeyFrame::mnImgWidth = 0;
    int KeyFrame::mnImgHeight = 0;

    int KeyFrame::mnGridRows = 0;
    int KeyFrame::mnGridCols = 0;

    KeyFrame::KeyFrame(const Frame &frame, Map *pMap) : mnReferenceFrameId(frame.mnId), N(frame.N),
                                                        mvKeysLeft(frame.mvKeysLeft), mvKeysRight(frame.mvKeysRight),
                                                        mvKeysRightWithSubPixel(frame.mvKeysRightWithSubPixel),
                                                        mDescriptorsLeft(frame.mDescriptorsLeft.clone()),
                                                        mDescriptorsRight(mDescriptorsRight.clone()),
                                                        mvMatches(frame.mvMatches), mvuRight(frame.mvuRight),
                                                        mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
                                                        mfLogScaleFactor(frame.mfLogScaleFactor),
                                                        mvScaleFactors(frame.mvScaleFactors),
                                                        mvLogScaleFactors(frame.mvLogScaleFactors),
                                                        mvInvScaleFactors(frame.mvInvScaleFactors),
                                                        mvLevelSigma2(frame.mvLevelSigma2),
                                                        mvInvLevelSigma2(frame.mvInvLevelSigma2),
                                                        mnLocalBAForKF(0), mnLocalBAForFixedKF(0)
    {
        if (!mbInitialization)
        {
            fx = GlobalParameters::fx;
            fy = GlobalParameters::fy;
            invfx = 1.0 / fx;
            invfy = 1.0 / fy;
            cx = GlobalParameters::cx;
            cy = GlobalParameters::cy;
            mb = GlobalParameters::mb;
            mbf = GlobalParameters::mbf;

            mnImgWidth = Frame::mnImgWidth;
            mnImgHeight = Frame::mnImgHeight;

            mnGridCols = GlobalParameters::mnGridCols;
            mnGridRows = GlobalParameters::mnGridRows;

            mbInitialization = true;
        }

        // KeyFrame id
        mnId = mnNext++;

        // Pose initialization
        SetPose(frame.mTcw);

        mvpMapPoints.resize(N, static_cast<MapPoint*>(NULL));

        // Global Map
        mpMap = pMap;
    }

    KeyFrame::~KeyFrame()
    {}

    void KeyFrame::SetPose(const cv::Mat &pose)
    {
        mTcw = pose.clone();
//
//        LOG(INFO) << "mTcw: " << mTcw;

        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mtcw = mTcw.rowRange(0, 3).col(3);

        mOw = -mRcw.t() * mtcw;


        mTwc = cv::Mat::eye(4, 4, mTcw.type());

        cv::Mat mRwc = mRcw.t();

        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));

        // UpdatePoseMatrix();
    }

    cv::Mat KeyFrame::GetPose() const
    {
        return mTcw.clone();
    }

    void KeyFrame::UpdatePoseMatrix()
    {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3).clone();
        mtcw = mTcw.rowRange(0, 3).col(3).clone();

        mOw = -mRcw.t() * mtcw;

        if (mTwc.empty())
            mTwc = cv::Mat::eye(4, 4, CV_32F);

        cv::Mat mRwc = mRwc.t();

        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));
    }

    void KeyFrame::AddMapPoint(MapPoint *pMP, const int &idx)
    {
        mvpMapPoints[idx] = pMP;
    }

    void KeyFrame::EraseMapPoint(MapPoint *pMP)
    {
        const int idx = pMP->GetIndexInKeyFrame(this);
        if (idx < 0) return;

        mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
    }

    void KeyFrame::EraseMapPointByIndex(const int &idx)
    {
        mvpMapPoints[idx] = static_cast<MapPoint*>(NULL);
    }

    void KeyFrame::AddConnection(KeyFrame *pKF, const int &w)
    {
        if (!pKF) return;

        mConnectedKeyFramesWithWeights[pKF] = w;

        UpdateConnections();
        UpdateBestCovisibleKeyFrames();
    }

    void KeyFrame::EraseConnection(KeyFrame *pKF)
    {
        if (!mConnectedKeyFramesWithWeights.count(pKF)) return;
        mConnectedKeyFramesWithWeights.erase(pKF);
    }

    std::map<KeyFrame*, int> KeyFrame::GetAllConnectedKeyFrames() const
    {
        return mConnectedKeyFramesWithWeights;
    }

    std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N) const
    {
        if (mvOrderedNeighbors.size() <= N)
            return mvOrderedNeighbors;

        return std::vector<KeyFrame*>(mvOrderedNeighbors.begin(), mvOrderedNeighbors.begin() + N);
    }

    std::vector<KeyFrame*> KeyFrame::GetCovisibilityKeyFramesByWeight(const int &w) const
    {
        std::vector<int>::const_iterator up = std::upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w);

        int pos = up - mvOrderedWeights.begin();
        return std::vector<KeyFrame*>(mvOrderedNeighbors.begin(), mvOrderedNeighbors.begin() + pos);
    }

    int KeyFrame::GetWeight(KeyFrame *pKF)
    {
        if (!pKF) return 0;
        if (!mConnectedKeyFramesWithWeights.count(pKF)) return 0;

        return mConnectedKeyFramesWithWeights[pKF];
    }

    void KeyFrame::UpdateConnections()
    {
        std::map<KeyFrame*, int> mKFCounter;

        for (std::vector<MapPoint*>::const_iterator vit = mvpMapPoints.begin(), vend = mvpMapPoints.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if (!pMP) continue;

            std::map<KeyFrame*, int> obs = pMP->GetAllObservations();
            if (obs.empty()) continue;

            for (std::map<KeyFrame*, int>::const_iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
            {
                if (mit->first->mnId == mnId)
                    continue;
                mKFCounter[mit->first]++;
            }

            if (mKFCounter.empty())
                return;
        }

       // if the counter greater than threshold then add connection
        int nmax = 0;
        KeyFrame* pKFmax = NULL;

        int th = 25;
        std::vector<std::pair<int, KeyFrame*> > vPairs;
        vPairs.reserve(mKFCounter.size());
        for (std::map<KeyFrame*, int>::const_iterator mit = mKFCounter.begin(), mend = mKFCounter.end(); mit != mend; mit++)
        {
            if (mit->second > nmax)
            {
                nmax = mit->second;
                pKFmax = mit->first;
            }

            if (mit->second > th)
            {
                vPairs.push_back(std::make_pair(mit->second, mit->first));
                (mit->first)->AddConnection(this, mit->second);
            }
        }

        if (vPairs.empty())
        {
            vPairs.push_back(std::make_pair(nmax, pKFmax));
            pKFmax->AddConnection(this, nmax);
        }

        std::sort(vPairs.begin(), vPairs.end());
        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;

        for (int i = 0, iend = vPairs.size(); i < iend; ++i)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }


        // Update related variable values
        mConnectedKeyFramesWithWeights = mKFCounter;
        mvOrderedNeighbors = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
    }

    std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
    {
        return mvOrderedNeighbors;
    }

    void KeyFrame::UpdateBestCovisibleKeyFrames()
    {
        std::vector<std::pair<int, KeyFrame*> > vWeightsAndNeighbors;
        vWeightsAndNeighbors.reserve(mConnectedKeyFramesWithWeights.size());

        for (std::map<KeyFrame*, int>::const_iterator mit = mConnectedKeyFramesWithWeights.begin(), mend = mConnectedKeyFramesWithWeights.end(); mit != mend; mit++)
        {
            vWeightsAndNeighbors.push_back(std::make_pair(mit->second, mit->first));
        }

        std::sort(vWeightsAndNeighbors.begin(), vWeightsAndNeighbors.end());

        std::list<KeyFrame*> lKFs;
        std::list<int> lWs;

        for (int i = 0, iend = vWeightsAndNeighbors.size(); i < iend; ++i)
        {
            lKFs.push_front(vWeightsAndNeighbors[i].second);
            lWs.push_front(vWeightsAndNeighbors[i].first);
        }

        mvOrderedNeighbors = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
        mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
    }

    std::vector<MapPoint*> KeyFrame::GetMapPointMatches() const
    {
       std::vector<MapPoint*> vMPs;
        vMPs.reserve(N);

        for (int i = 0; i < N; ++i)
        {
            MapPoint* pMP = mvpMapPoints[i];
            if (!pMP) continue;

            vMPs.push_back(pMP);
        }

        return vMPs;
    }

    cv::Mat KeyFrame::GetCameraCenter() const
    {
        return mOw.clone();
    }

    cv::Mat KeyFrame::GetPoseInverse() const
    {
        return mTwc.clone();
    }


}

