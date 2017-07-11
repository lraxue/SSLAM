//
// Created by feixue on 17-6-12.
//

#ifndef SSLAM_MAPPOINT_H
#define SSLAM_MAPPOINT_H

#include <Frame.h>
#include <KeyFrame.h>
#include <EpipolarTriangle.h>

#include <opencv2/opencv.hpp>

namespace SSLAM
{
    class Frame;
    class KeyFrame;
    class Map;
    class EpipolarTriangle;

    class MapPoint
    {
    public:
        MapPoint(Map* pMap, const Frame& frame, const int& idx);
        ~MapPoint();

    public:
        // Position functions
        void SetPos(const cv::Mat& pos);
        cv::Mat GetPos();

        cv::Mat GetDescriptor();

        // Observations
        void AddObservation(KeyFrame* pKF, const int& idx);
        void EraseObservation(KeyFrame* pKF);
        std::map<KeyFrame*, int> GetAllObservations() const;
        int GetIndexInKeyFrame(KeyFrame* pKF);
        int Observations();

        // Triangles
        void AddTriangle(EpipolarTriangle* pTriangle);
        void EraseTriangle(EpipolarTriangle* pTriangle);
        std::vector<EpipolarTriangle*> GetAllTriangles() const;
        EpipolarTriangle* GetLastEpipolarTriangle();
        EpipolarTriangle* GetFirstEpipolarTriangle();
        int Triangles();



        void AddFounder(const unsigned long& frameID, const int& idx);

        // Update descriptor and normal vector
        void UpdateNormalAndDepth();

        cv::Mat GetNormal();

        void ComputeDistinctiveDescriptors();

        float GetMinDistanceInvariance();

        float GetMaxDistanceInvariance();

        int PredictScale(const float& currentDist, KeyFrame* pKF);
        int PredictScale(const float& currentDist, Frame* pF);

        void SetBad();

        bool IsBad();

        bool IsInKeyFrame(KeyFrame* pKF) const;

        static bool lId(MapPoint* pMP1, MapPoint* pMP2)
        {
            return pMP1->mnId < pMP2->mnId;
        }

        // Replace
        void Replace(MapPoint* pMP);

    public:
        unsigned long mnId;
        bool mbBad;   // Tag for discarding

        int nObs;
        KeyFrame* mpRefKF;
        MapPoint* mpReplaced;

        // Tags for bundle adjustment
        unsigned long mnLocalBAForKF;

        // Variables used by the tracking
        float mTrackProjX;
        float mTrackProjY;
        float mTrackProjXR;
        bool mbTrackInView;
        int mnTrackScaleLevel;
        float mTrackViewCos;

        // Tags for tracking
        unsigned long mnTrackLocalMapForFrame;
        unsigned long mnLastFrameSeen;
        unsigned long mnFuseMapPointForKF;

    protected:
        // MapPoint Id

        static unsigned long mnNext;

        // World coordinate
        cv::Mat mPos;  // 3x1

        // Descriptor
        cv::Mat mDescriptor;

        // Feature flow
        // Frame id and index in frame
        std::map<unsigned long, int> mFeatureFlow;
        std::map<KeyFrame*, int> mObservations;

        // Triangle flow along time line
        std::list<EpipolarTriangle*> mlpTriangles;

        cv::Mat mNormalVector;

        // Scale invariance distance
        float mfMinDistance;
        float mfMaxDistance;

        // Global Map
        Map* mpMap;


    };
}

#endif //SSLAM_MAPPOINT_H
