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
        EpipolarTriangle(const cv::Mat& normal, const float& d);
        EpipolarTriangle(const cv::Point3f& p1, const cv::Point3f& p2, const cv::Point3f& p3);

        ~EpipolarTriangle();

    public:
        cv::Mat GetNormal() const;

        cv::Mat GetRawNormal() const;

        float GetDistance() const ;

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

    public:
        unsigned long mnId;
        static unsigned long mnNext;

    protected:
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
