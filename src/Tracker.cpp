//
// Created by feixue on 17-6-13.
//

#include <Tracker.h>
#include <ORBmatcher.h>
#include <Optimizer.h>
#include <EpipolarTriangle.h>
#include <Monitor.h>
#include <GlobalParameters.h>

#include <Test.h>


#include <glog/logging.h>

using namespace std;

namespace SSLAM
{
    Tracker::Tracker(FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, LocalMapper *pLocalMapper, Map *pMap) :
            mpMap(pMap), mpFrameDrawer(pFrameDrawer),
            mpMapDrawer(pMapDrawer), mpLocalMapper(pLocalMapper),
            mState(NOT_INITIALIZED)
    {

        mVelocity = cv::Mat::eye(4, 4, CV_32F);

        mThDepth = GlobalParameters::mThDepth;
    }

    Tracker::~Tracker()
    {
    }

    cv::Mat Tracker::GrabStereo(const cv::Mat &imLeft, const cv::Mat &imRight)
    {
        mCurrentFrame = Frame(imLeft, imRight);

//        if (mCurrentFrame.mnId > 0)
//        {
//            // Test
//            Test::TestMatch(mLastFrame, mCurrentFrame);
//        }

        Track();

        return mCurrentFrame.mTcw;
    }

    void Tracker::Track()
    {
        bool bOK;
        if (mState == NOT_INITIALIZED)
        {
            bOK = StereoInitialization();

            mpFrameDrawer->Update(this);

            if (bOK)
                mState = SUCCEED;
        }
        else
        {
            // Track frame after initialization
            bOK = TrackBasedOnMotionPrediction();

            bOK = TrackLocalMap();

            LOG(INFO) << "Frame: " << mCurrentFrame.mnId << " " << mCurrentFrame.mTcw;

            mpFrameDrawer->Update(this);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat::eye(4, 4, CV_32F);

            // According to tacking information, update current KeyFrame
            UpdateCurrentFrame();

            // Each Frame associate to one KeyFrame
            // CreateNewKeyFrame();
            CreateNewKeyFrameBasedOnTrackAbility();
        }

        // Information transfer
        mLastFrame = Frame(mCurrentFrame);
    }

    bool Tracker::TrackBasedOnMotionPrediction()
    {
        // Predicted motion
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        ORBmatcher matcher(0.8, true);
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        int th = 7;
        int nMatchedPoints = matcher.SearchCircleMatchesByProjection(mLastFrame, mCurrentFrame, th);
        LOG(INFO) << "Number of tracked points: " << nMatchedPoints;

        if (nMatchedPoints < 20)
        {
            std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
                      static_cast<MapPoint *>(NULL));
            nMatchedPoints = matcher.SearchCircleMatchesByProjection(mLastFrame, mCurrentFrame, th * 2);
        }

        LOG(INFO) << "Reprojection error before optimization: " << mCurrentFrame.ComputeReprojectionError();

        // Number of inliers after optimization
        int nInliers = 0;

        // Use Epipolar triangle for optimization

//#ifdef USE_TRIANGLE
//        nInliers = Optimizer::OptimisePoseOn3DPoints(mCurrentFrame);
//#else
//        nInliers = Optimizer::OptimizePose(mCurrentFrame);
//#endif

        nInliers = Optimizer::OptimizePose(mCurrentFrame);

        int nMatchesInMap = 0;
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutliers[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutliers[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nMatchedPoints--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                {
                    nMatchesInMap++;
                }
            }
        }


        // For debugging
//        cv::Mat mReprojAfterOptim;
//        std::vector<cv::KeyPoint> vKeysLeft;
//        std::vector<cv::KeyPoint> vKeysRight;
//        std::vector<cv::Point2f> vProjPointsLeft;
//        std::vector<cv::Point2f> vProjPointsRight;
//        std::vector<bool> vBInliers;
//
//        std::vector<int>& vMatches = mCurrentFrame.mvMatches;
//        for (int i = 0; i < mCurrentFrame.N; ++i)
//        {
//            if (!mCurrentFrame.mvpMapPoints[i]) continue;
//            vKeysLeft.push_back(mCurrentFrame.mvKeysLeft[i]);
//            vKeysRight.push_back(mCurrentFrame.mvKeysRight[vMatches[i]]);
//
////            cv::Point2f proPL = mCurrentFrame.Project3DPointOnLeftImage(i);
////            cv::Point2f proPR = proPL;
//            cv::Point2f proPL = mCurrentFrame.mvKeysLeft[i].pt;
//            cv::Point2f proPR = mCurrentFrame.mvKeysRight[vMatches[i]].pt;
//            proPR.x = mCurrentFrame.mvuRight[i];
//            // mCurrentFrame.mbf / mCurrentFrame.mvDepth[i];
//
//            vProjPointsLeft.push_back(proPL);
//            vProjPointsRight.push_back(proPR);
//
//            if (mCurrentFrame.mvbOutliers[i]) vBInliers.push_back(false);
//            else vBInliers.push_back(true);
//        }
//
//        Monitor::DrawReprojectedPointsOnStereoFrameAfterOptimization(mCurrentFrame.mRGBLeft, mCurrentFrame.mRGBRight,
//                                                                     vKeysLeft, vKeysRight, vProjPointsLeft, vProjPointsRight,
//                                                                     vBInliers, mReprojAfterOptim);
//
//        const std::string tag = "Frame: " + std::to_string(mCurrentFrame.mnId) + "-";
//        cv::imshow("pro image after opt", mReprojAfterOptim);
//
////        cv::imwrite(tag + "pro image after opt.png", mReprojAfterOptim);
//        cv::waitKey(0);

        //**************** end of debugging*************

        LOG(INFO) << "Number of inliers after tracking based on motion model: " << nMatchedPoints;
        LOG(INFO) << "Reprojection error after tracking based on motion model: " <<  mCurrentFrame.ComputeReprojectionError();

        return nMatchesInMap >= 10;

    }

    bool Tracker::TrackLocalMap()
    {
        UpdateLocalMap();

        SearchLocalMapPoints();

        int nInliers = 0;
        // Optimize pose

//#ifdef USE_TRIANGLE
//        nInliers = Optimizer::OptimisePoseOn3DPoints(mCurrentFrame);
//#else
//        nInliers = Optimizer::OptimizePose(mCurrentFrame);
//#endif

        nInliers = Optimizer::OptimizePose(mCurrentFrame);
        int mnMatchesInliers = 0;

        // Update MapPoints
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP || pMP->IsBad()) continue;

            if (!mCurrentFrame.mvbOutliers[i])
            {
                mnMatchesInliers++;
            }
            else
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }

        LOG(INFO) << "Reprojection error after track local map " << mCurrentFrame.ComputeReprojectionError();
//        // For debugging
//        cv::Mat mReprojAfterOptim;
//        std::vector<cv::KeyPoint> vKeysLeft;
//        std::vector<cv::KeyPoint> vKeysRight;
//        std::vector<cv::Point2f> vProjPointsLeft;
//        std::vector<cv::Point2f> vProjPointsRight;
//        std::vector<bool> vBInliers;
//
//        std::vector<int>& vMatches = mCurrentFrame.mvMatches;
//        for (int i = 0; i < mCurrentFrame.N; ++i)
//        {
//            if (!mCurrentFrame.mvpMapPoints[i]) continue;
//            vKeysLeft.push_back(mCurrentFrame.mvKeysLeft[i]);
//            vKeysRight.push_back(mCurrentFrame.mvKeysRight[vMatches[i]]);
//
////            cv::Point2f proPL = mCurrentFrame.Project3DPointOnLeftImage(i);
////            cv::Point2f proPR = proPL;
//            cv::Point2f proPL = mCurrentFrame.mvKeysLeft[i].pt;
//            cv::Point2f proPR = mCurrentFrame.mvKeysRight[vMatches[i]].pt;
//            proPR.x = mCurrentFrame.mvuRight[i];
//            // mCurrentFrame.mbf / mCurrentFrame.mvDepth[i];
//
//            vProjPointsLeft.push_back(proPL);
//            vProjPointsRight.push_back(proPR);
//
//            if (mCurrentFrame.mvbOutliers[i]) vBInliers.push_back(false);
//            else vBInliers.push_back(true);
//        }
//
//        Monitor::DrawReprojectedPointsOnStereoFrameAfterOptimization(mCurrentFrame.mRGBLeft, mCurrentFrame.mRGBRight,
//                                                                     vKeysLeft, vKeysRight, vProjPointsLeft, vProjPointsRight,
//                                                                     vBInliers, mReprojAfterOptim);
//
//        const std::string tag = "Frame: " + std::to_string(mCurrentFrame.mnId) + "-";
//        cv::imshow("pro image after opt", mReprojAfterOptim);
//
////        cv::imwrite(tag + "localmap pro image after opt.png", mReprojAfterOptim);
//        cv::waitKey(0);

        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
    }

    void Tracker::UpdateLastFrame()
    {
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        mLastFrame.SetPose(mpReferenceKF->GetPose());

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++)
        {
            float z = mLastFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (vDepthIdx.empty())
            return;

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
            }

            if (bCreateNew)
            {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(mpMap, mLastFrame, i);
                pNewMP->SetPos(x3D);

                mLastFrame.mvpMapPoints[i] = pNewMP;

//                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }
    }

    void Tracker::UpdateCurrentFrame()
    {

        std::vector<std::pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
                vDepthIdx.push_back(std::make_pair(z, i));
        }

        std::sort(vDepthIdx.begin(), vDepthIdx.end());


        int nTrackedGlobalMPs = 0;
        int nTrackedTemporalMPs = 0;
        int nCreatedTemporalMPs = 0;

        bool bCreatedNew = false;

        for (int j = 0; j < vDepthIdx.size(); ++j)
        {
            int i = vDepthIdx[j].second;
            bCreatedNew = false;

            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP || pMP->IsBad())
                bCreatedNew = true;
            if (mCurrentFrame.mvbOutliers[i])
                bCreatedNew;

            if (vDepthIdx[i].first < 0 || vDepthIdx[j].first > 3 * mThDepth)
                continue;

            if (bCreatedNew)
            {
                cv::Mat X3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(mpMap, mCurrentFrame, i);
                pNewMP->SetPos(X3D);

                // Associate EpipolarTriangle
                EpipolarTriangle *pNewTriangle;

#ifdef USE_TRIANGLE
                pNewTriangle = mCurrentFrame.mvpTriangles[i];
#else
                pNewTriangle = mCurrentFrame.GenerateEpipolarTriangle(i);
                mCurrentFrame.mvpTriangles[i] = pNewTriangle;
#endif

                pNewMP->AddTriangle(pNewTriangle);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                nCreatedTemporalMPs++;
            }
            else
            {
                if (pMP->mType == MapPoint::mGlobalPoint)
                    nTrackedGlobalMPs++;
                else if (pMP->mType == MapPoint::mTemporalPoint)
                    nTrackedTemporalMPs++;


                pMP->AddFounder(mCurrentFrame.mnId, i);

                // Associate EpipolarTriangle
                EpipolarTriangle *pNewTriangle;

#ifdef USE_TRIANGLE
                pNewTriangle = mCurrentFrame.mvpTriangles[i];
#else
                pNewTriangle = mCurrentFrame.GenerateEpipolarTriangle(i);
                mCurrentFrame.mvpTriangles[i] = pNewTriangle;
#endif

                pMP->AddTriangle(pNewTriangle);
            }
        }

        LOG(INFO) << "Update Frame: " << mCurrentFrame.mnId
                  << " with tracked global MPs: " << nTrackedGlobalMPs
                  << ", tracked temporal MPs: " << nTrackedTemporalMPs
                  << ", created temporal MPs:" << nCreatedTemporalMPs;
    }


    void Tracker::UpdateLocalKeyFrames()
    {
        // Use the observed MapPoints search related KeyFrames
        std::map<KeyFrame *, int> KFCounter;
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP) continue;

            const std::map<KeyFrame *, int> obs = pMP->GetAllObservations();
            for (std::map<KeyFrame *, int>::const_iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
            {
                KFCounter[mit->first]++;
            }

            if (KFCounter.empty())
                return;

            mvpLocalKeyFrames.clear();

            for (std::map<KeyFrame *, int>::const_iterator mit = KFCounter.begin(), mend = KFCounter.end();
                 mit != mend; mit++)
            {
                mvpLocalKeyFrames.push_back(mit->first);
            }
        }

        LOG(INFO) << "Number of local KeyFrames: " << mvpLocalKeyFrames.size();

    }

    void Tracker::UpdateLocalMapPoints()
    {
        mvpLocalMapPoints.clear();

        for (std::vector<KeyFrame *>::const_iterator vitKF = mvpLocalKeyFrames.begin(), vendKF = mvpLocalKeyFrames.end();
             vitKF != vendKF; vitKF++)
        {
            KeyFrame *pKF = *vitKF;

            const std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (std::vector<MapPoint *>::const_iterator vitMP = vpMPs.begin(), vendMP = vpMPs.end();
                 vitMP != vendMP; vitMP++)
            {
                MapPoint *pMP = *vitMP;
                if (!pMP)
                    continue;
                if (pMP->IsBad())
                    continue;

                if (pMP->mnTrackLocalMapForFrame == mCurrentFrame.mnId)
                    continue;

                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackLocalMapForFrame = mCurrentFrame.mnId;
            }
        }

        LOG(INFO) << "Number of local MapPoints: " << mvpLocalMapPoints.size();
    }

    void Tracker::UpdateLocalMap()
    {
        UpdateLocalKeyFrames();
        UpdateLocalMapPoints();
    }

    void Tracker::SearchLocalMapPoints()
    {
        for (std::vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++)
        {
            MapPoint *pMP = *vit;

            if (!pMP)
                continue;

            pMP->mnLastFrameSeen = mCurrentFrame.mnId;
            pMP->mbTrackInView = false;
        }

        int nToMatch = 0;

        for (std::vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;

            // Project
            if (mCurrentFrame.IsInFrustum(pMP, 0.5))
                nToMatch++;
        }

        // Project points in frame
        ORBmatcher matcher(0.8);
        int th = 5;
        int nMatches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);

        LOG(INFO) << "ToMatch: " << nToMatch << " ,nMatches: " << nMatches;
    }

    void Tracker::CreateNewKeyFrame()
    {
        KeyFrame *pNewKF = new KeyFrame(mCurrentFrame, mpMap);
//        mpMap->AddKeyFrame(pNewKF);


        mpReferenceKF = pNewKF;

        int nPoints = 0;

        // mCurrentFrame.UpdatePoseMatrix();
        std::vector<std::pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
                vDepthIdx.push_back(std::make_pair(z, i));
        }

        std::sort(vDepthIdx.begin(), vDepthIdx.end());

        int nNewCreatedMapPoints = 0;

        for (int j = 0; j < vDepthIdx.size(); ++j)
        {
            int i = vDepthIdx[j].second;
            bool bCreateNew = false;

            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

            if (mCurrentFrame.mvDepth[i] > mThDepth * 3) continue;

            if (!pMP || pMP->IsBad())
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
            else   // This MapPoint is associated to a feature-pair in current frame
            {
                pNewKF->AddMapPoint(pMP, i);

                EpipolarTriangle *pNewTriangle;

#ifdef USE_TRIANGLE
                pNewTriangle = mCurrentFrame.mvpTriangles[i];
#else
                pNewTriangle = mCurrentFrame.GenerateEpipolarTriangle(i);
                mCurrentFrame.mvpTriangles[i] = pNewTriangle;
#endif

                // Associate EpipolarTriangle
                pMP->AddTriangle(pNewTriangle);

                nPoints++;
            }


            if (bCreateNew && nNewCreatedMapPoints < 300)
            {
                cv::Mat X3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(mpMap, mCurrentFrame, i);
                pNewMP->mpRefKF = pNewKF;
                pNewMP->SetPos(X3D);

                pNewKF->AddMapPoint(pNewMP, i);
                pNewMP->AddObservation(pNewKF, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;

                // Associate EpipolarTriangle
                EpipolarTriangle *pNewTriangle;

#ifdef USE_TRIANGLE
                pNewTriangle = mCurrentFrame.mvpTriangles[i];
#else
                pNewTriangle = mCurrentFrame.GenerateEpipolarTriangle(i);
                mCurrentFrame.mvpTriangles[i] = pNewTriangle;
#endif

                pNewMP->AddTriangle(pNewTriangle);


                nNewCreatedMapPoints++;

                nPoints++;
            }

//            if (nPoints > 300) break;
        }

        // First, process new created KeyFrame
        mpLocalMapper->ProcessNewKeyFrame(pNewKF);

        // Then, add new created KeyFrame to global Map
        mpMap->AddKeyFrame(pNewKF);

        LOG(INFO) << "Create new KeyFrame: " << pNewKF->mnId << " based on Frame:" << mCurrentFrame.mnId;
        LOG(INFO) << "Create new MapPoints: " << nNewCreatedMapPoints << " ,total points: " << nPoints;
    }


    void Tracker::CreateNewKeyFrameBasedOnTrackAbility()
    {
        KeyFrame *pNewKF = new KeyFrame(mCurrentFrame, mpMap);

        mpReferenceKF = pNewKF;

        int nOldGlobalMPs = 0;
        int nNewGlobalMPs = 0;

        const int thAge = 3;
        bool bUpgradeMPBasedOnAge = pNewKF->mnId > 5 ? true:false;

        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP) continue;

            // For tracked global MapPoint
            if (pMP->mType == MapPoint::mGlobalPoint)
            {
                pNewKF->AddMapPoint(pMP, i);

                nOldGlobalMPs++;
            }
            else // For temporal MapPoint
            {
                if (bUpgradeMPBasedOnAge)
                {
                    if (pMP->GetAge() > thAge)
                    {
                        pMP->mType = MapPoint::mGlobalPoint;

                        pMP->mpRefKF = pNewKF;
                        pNewKF->AddMapPoint(pMP, i);
//                        pMP->ComputeDistinctiveDescriptors();
//                        pMP->UpdateNormalAndDepth();

                        mpMap->AddMapPoint(pMP);

                        nNewGlobalMPs++;
                    }
                }
                else
                {
                    if (mCurrentFrame.mvDepth[i] < mThDepth)
                    {
                        pMP->mType = MapPoint::mGlobalPoint;

                        pMP->mpRefKF = pNewKF;
                        pNewKF->AddMapPoint(pMP, i);

//                        pMP->ComputeDistinctiveDescriptors();
//                        pMP->UpdateNormalAndDepth();

                        mpMap->AddMapPoint(pMP);
                        nNewGlobalMPs++;
                    }

                }
            }
        }

        LOG(INFO) << "Create new KeyFrame: " << pNewKF->mnId << " based on Frame:" << mCurrentFrame.mnId;
        LOG(INFO) << "KeyFrame: " << pNewKF->mnId << " with old global MPs: " << nOldGlobalMPs << " , new global MPs: " << nNewGlobalMPs;

        // Process new created KeyFrame
        mpLocalMapper->ProcessNewKeyFrame(pNewKF);

        // Add new KeyFrame to the global Map
        mpMap->AddKeyFrame(pNewKF);


//        LOG(INFO) << "Create new KeyFrame: " << pNewKF->mnId << " based on Frame:" << mCurrentFrame.mnId;
//        LOG(INFO) << "KeyFrame: " << pNewKF->mnId << " with old global MPs: " << nOldGlobalMPs << " , new global MPs: " << nNewGlobalMPs;
    }

    bool Tracker::StereoInitialization()
    {
        if (mCurrentFrame.N < 100)
            return false;

        // The original position
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        // Create the first KeyFrame
        KeyFrame *pNewKF = new KeyFrame(mCurrentFrame, mpMap);
        mpMap->AddKeyFrame(pNewKF);

        int nNewCreatedMapPoints = 0;
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            const float z = mCurrentFrame.mvDepth[i];
            if (z > 0 && z < mThDepth)
            {
                const cv::Mat X3D = mCurrentFrame.UnprojectStereo(i);

                MapPoint *pNewMP = new MapPoint(mpMap, mCurrentFrame, i);
                pNewMP->mpRefKF = pNewKF;

                pNewMP->SetPos(X3D);
                pNewMP->mType = MapPoint::mGlobalPoint;

                mCurrentFrame.mvpMapPoints[i] = pNewMP;

                pNewKF->AddMapPoint(pNewMP, i);
                pNewMP->AddObservation(pNewKF, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pNewMP);
                nNewCreatedMapPoints++;

                // Associate EpipolarTriangle
                EpipolarTriangle *pNewTriangle;

#ifdef USE_TRIANGLE
                pNewTriangle = mCurrentFrame.mvpTriangles[i];
#else
                pNewTriangle = mCurrentFrame.GenerateEpipolarTriangle(i);
                mCurrentFrame.mvpTriangles[i] = pNewTriangle;
#endif

                pNewMP->AddTriangle(pNewTriangle);

            }
        }

        mpReferenceKF = pNewKF;

        LOG(INFO) << "Create new KeyFrame: " << pNewKF->mnId << " based on Frame: " << mCurrentFrame.mnId;
        LOG(INFO) << "New created " << nNewCreatedMapPoints << " MapPoints in from Frame: " << mCurrentFrame.mnId;

        return true;
    }
}