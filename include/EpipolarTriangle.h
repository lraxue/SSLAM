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
        // Constructor functions
        EpipolarTriangle();
        EpipolarTriangle(const unsigned long& frameId, const cv::Mat& normal, const float& d);
        EpipolarTriangle(const unsigned long& frameId, const cv::Mat& X, const cv::Mat& Cl, const cv::Mat& Cr);

        ~EpipolarTriangle();

    public:
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

    protected:
        /**
         * Compute three inner angles, as the structure of the triangle
         */
        void ComputeThreeAngles();


    public:
        unsigned long mnId;
        static unsigned long mnNext;

        // Id of the Frame that creating this ETriangle
        unsigned long mnFrameId;

    protected:
        // Attributes
        float mAngle1, mAngle2, mAngle3;  // Three angles: top-left-right

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
