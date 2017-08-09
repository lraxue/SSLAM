//
// Created by feixue on 17-6-21.
//

#ifndef SSLAM_TRIANGLE_H
#define SSLAM_TRIANGLE_H

#include <opencv2/opencv.hpp>

namespace SSLAM
{
    class EpipolarTriangle
    {
    public:
        struct SUncertainty
        {
            float mResponseLeft;
            float mResponseRight;
            float mMatchCost;

            // Constructor
            SUncertainty(const float ResponseLeft, const float& ResponseRight, const float& MatchCost):
                    mResponseLeft(ResponseLeft), mResponseRight(ResponseRight), mMatchCost(MatchCost)
            {}

            SUncertainty():mResponseLeft(0.f), mResponseRight(0.f), mMatchCost(0.f)
            {}

            // Deep copy
            SUncertainty(const SUncertainty& Uncertainty)
            {
                this->mResponseLeft = Uncertainty.mResponseLeft;
                this->mResponseRight = Uncertainty.mResponseRight;
                this->mMatchCost = Uncertainty.mMatchCost;
            }

            float IntrinsicUncertainty()
            {
                return std::exp(mResponseLeft / 255.f + mResponseRight / 255.f - mMatchCost / (255 * 25.f));
            }
        };

    public:
        // Constructor functions
        EpipolarTriangle();
        EpipolarTriangle(const unsigned long& frameId, const cv::Mat& normal, const float& d);
        EpipolarTriangle(const unsigned long& frameId, const cv::Mat& X, const cv::Mat& Cl, const cv::Mat& Cr);

        // Extended constructor functions
        EpipolarTriangle(const unsigned long& frameId, const cv::Mat& X, const cv::Mat& Cl, const cv::Mat& Cr,
                         const cv::KeyPoint& keyLeft, const cv::KeyPoint& keyRight, const float& depth, const float& uRight);

        // Extended Constructor functions
        EpipolarTriangle(const unsigned long& frameId, const cv::Mat& X, const cv::Mat& Cl, cv::Mat& Cr, const SUncertainty& uncertainty);

        ~EpipolarTriangle();

    public:
        cv::Mat GetX() const;

        cv::Mat GetNormal() const;

        cv::Mat GetRawNormal() const;

        float GetDistance() const;

        float GetRawDistance() const;

        void SetNormalAndDistance(const cv::Mat& normal, const float& d);

        void Transform(const cv::Mat& R, const cv::Mat& t);

        void Transform(const cv::Mat& T);

        /**
         *
         * @param x3Dw a 3D point in world coordinate
         * @return distance to the plane
         */
        float ComputeDistance(const cv::Mat& x3Dw);

        /**
         *
         * @param idx point id of three points
         * @return distance
         */
        float ComputeDistanceToVertex(const cv::Mat& x3Dw, const int& idx = 0);

        // Get three angles of the triangle
        float Angle1() const;
        float Angle2() const;
        float Angle3() const;


        // Get uncertainty
        float Uncertainty() const;

        // Coordinate in left and right image
        cv::Point2f PointLeft() const;
        cv::Point2f PointRight() const;

        // Descriptor in left and right image
        cv::Mat DescriptorLeft() const;
        cv::Mat DescriptorRight() const;

    protected:
        /**
         * Compute three inner angles, as the structure of the triangle
         */
        void ComputeThreeAngles();

        /**
         * Compute normal and distance
         */

        void ComputeNormalAndDistance();

        void ComputeUncertainty();


    public:
        unsigned long mnId;
        static unsigned long mnNext;

        // Id of the Frame that creating this ETriangle
        unsigned long mnFrameId;

        // Uncertainty
        SUncertainty mUncertainty;

    protected:
        // Basic information
        cv::KeyPoint mKeyLeft, mKeyRight;
        cv::Mat mDescLeft, mDescRight;
        float mDepth;
        float muRight;

        // Uncertainty
        float mResponse;
        float mMatchRatio;
        float mAngleRatio;
        float mFusedUncertainty;

        // Attributes
        float mAngle1, mAngle2, mAngle3;  // Three angles: top-left-right

        // Uncertainty defined on observation
        float mObservation;

        // Parameters
        static float mMatchTheta;
        static float mAngleTheta;
        static float alphaF, alphaA, alphaM;
        static bool bInitialization;


        // Parameters after transformation
        cv::Mat mNormal;
        float mD;

        cv::Mat mRawNormal;
        float mRawD;

        // Three points that define the triangle
        cv::Mat mX;
        cv::Mat mCl;
        cv::Mat mCr;
    };
}

#endif //SSLAM_TRIANGLE_H
