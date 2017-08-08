//
// Created by feixue on 17-6-15.
//

#ifndef SSLAM_KEYFRAME_H
#define SSLAM_KEYFRAME_H

#include <Frame.h>
#include <MapPoint.h>
#include <Map.h>

namespace SSLAM
{
    class Frame;
    class MapPoint;
    class Map;

    class KeyFrame
    {
    public:
        KeyFrame(const Frame& frame, Map* pMap);
        ~KeyFrame();

        void SetPose(const cv::Mat& pose);
        cv::Mat GetPose() const;

        cv::Mat GetCameraCenter() const ;
        cv::Mat GetPoseInverse() const;
        cv::Mat GetRotation() const;
        cv::Mat GetTranslation() const;


        // MapPoint functions
        void AddMapPoint(MapPoint* pMP, const int& idx);
        void EraseMapPoint(MapPoint* pMP);
        void EraseMapPointByIndex(const int& idx);
        void ReplaceMapPoint(const int& idx, MapPoint* pMP);

        // Covisibility functions
        void AddConnection(KeyFrame* pKF, const int& w);
        void EraseConnection(KeyFrame* pKF);
        std::map<KeyFrame*, int> GetAllConnectedKeyFrames() const;
        std::vector<KeyFrame*> GetCovisibilityKeyFramesByWeight(const int& w) const;
        std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int& N) const;
        std::vector<MapPoint*> GetMapPointMatches();
        int GetWeight(KeyFrame* pKF);

        std::vector<KeyFrame*> GetVectorCovisibleKeyFrames();

        void UpdateConnections();
        void UpdateBestCovisibleKeyFrames();


        // Compute reprojection error
        cv::Point2f Project3DPointOnLeftImage(const int& idx);
        float ComputeReprojectionError(const int& idx);
        float ComputeReprojectionError(MapPoint* pMP);


        static bool lId(KeyFrame* pKF1, KeyFrame* pKF2)
        {
            return pKF1->mnId < pKF2->mnId;
        }

        static bool weightComp(int a, int b)
        {
            return a > b;
        }



        // Fusion functions
        bool IsInImage(const float& u, const float& v) const;
        std::vector<int> SearchFeaturesInGrid(const float& cX, const float& cY, const float& radius, const int minLevel=-1, const int maxLevel=-1) const ;
        MapPoint* GetMapPoint(const int& idx);




    protected:
        void UpdatePoseMatrix();


    public:
        // KeyFrame id
        unsigned long mnId;
        static unsigned long mnNext;

        unsigned long mnFuseMapPointsForKF;

        const int N;

        const std::vector<cv::KeyPoint> mvKeysLeft;
        const std::vector<cv::KeyPoint> mvKeysRight;
        const std::vector<cv::KeyPoint> mvKeysRightWithSubPixel;

        const cv::Mat mDescriptorsLeft;
        const cv::Mat mDescriptorsRight;

        const std::vector<int> mvMatches;  // index of keypoints in right image, -1 default.
        const std::vector<float> mvuRight;
        const std::vector<float> mvDepth;
        std::vector<bool> mvbOutliers;

        const int mnScaleLevels;
        const float mfScaleFactor;
        const float mfLogScaleFactor;
        const std::vector<float> mvScaleFactors;
        const std::vector<float> mvLogScaleFactors;
        const std::vector<float> mvInvScaleFactors;
        const std::vector<float> mvLevelSigma2;
        const std::vector<float> mvInvLevelSigma2;

        // Calibration parameters
        static float fx;
        static float fy;
        static float invfx;
        static float invfy;
        static float cx;
        static float cy;
        static float mb;
        static float mbf;
        static bool mbInitialization;

        // Grid size
        static int mnGridRows;
        static int mnGridCols;

        // Image size
        static int mnImgWidth;
        static int mnImgHeight;

        // Tas for bundle adjustment
        unsigned long mnLocalBAForKF;
        unsigned long mnLocalBAForFixedKF;

    protected:
        // Pose
        cv::Mat mTcw;
        cv::Mat mRcw;
        cv::Mat mtcw;
        cv::Mat mTwc;
        cv::Mat mOw;

        // MapPoints
        std::vector<MapPoint*> mvpMapPoints;

        // Map
        Map* mpMap;

        // Neighbors and weights
        std::map<KeyFrame*, int> mConnectedKeyFramesWithWeights;
        std::vector<int> mvOrderedWeights;
        std::vector<KeyFrame*> mvOrderedNeighbors;

        // Reference Frame id
        const unsigned long mnReferenceFrameId;

        // Features in grid
        std::vector<std::vector<std::vector<int> > >mvFeaturesInGrid;

    };
}

#endif //SSLAM_KEYFRAME_H
