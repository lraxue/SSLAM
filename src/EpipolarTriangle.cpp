//
// Created by feixue on 17-6-21.
//

#include <EpipolarTriangle.h>

namespace SSLAM
{
    unsigned long EpipolarTriangle::mnNext = 0;

    EpipolarTriangle::EpipolarTriangle()
    {}

    EpipolarTriangle::EpipolarTriangle(const cv::Mat &normal, const float &d) : mNormal(normal), mD(d)
    {
        mnId = mnNext++;
    }

    EpipolarTriangle::EpipolarTriangle(const cv::Point3f &p1, const cv::Point3f &p2, const cv::Point3f &p3)
    {
        mnId = mnNext++;

        // Initialize the three vertexes
        mX = (cv::Mat_<float>(3, 1) << p1.x, p1.y, p1.z);
        mCl = (cv::Mat_<float>(3, 1) << p2.x, p2.y, p2.z);
        mCr = (cv::Mat_<float>(3, 1) << p3.x, p3.y, p3.z);

        // Normal and distance
        cv::Mat X12 = mCl - mX;
        cv::Mat X13 = mCl - mCr;

        cv::Mat normal = X12.cross(X13);

        mNormal = normal / cv::norm(normal);
        mD = cv::norm(mX.t() * mNormal);

        mRawNormal = mNormal;
        mRawD = mD;

    }

    EpipolarTriangle::~EpipolarTriangle()
    {

    }

    cv::Mat EpipolarTriangle::GetNormal() const
    {
        return mNormal.clone();
    }

    cv::Mat EpipolarTriangle::GetRawNormal() const
    {
        return mRawNormal.clone();
    }

    float EpipolarTriangle::GetRawDistance() const
    {
        return mRawD;
    }

    float EpipolarTriangle::GetDistance() const
    {
        return mD;
    }

    void EpipolarTriangle::SetNormalAndDistance(const cv::Mat &normal, const float &d)
    {
        mNormal = normal.clone();
        mD = d;
    }


    float EpipolarTriangle::ComputeDistance(const cv::Mat &x3Dw)
    {
        return abs(x3Dw.dot(mNormal) + mD);
    }

    void EpipolarTriangle::Transform(const cv::Mat &T)
    {
        if (T.empty())
            return;

        // Extract R and t
        const cv::Mat R = T.rowRange(0, 3).colRange(0, 3).clone();
        const cv::Mat t = T.rowRange(0, 3).col(3);

        Transform(R, t);
    }

    void EpipolarTriangle::Transform(const cv::Mat &R, const cv::Mat &t)
    {
        // Update normal vector and distance to the origin
        // n' = R * n
        // d' = -t.t() * n' + d

        if (R.empty() || t.empty())
            return;

        mNormal = R * mRawNormal;
        mD = fabs(-mNormal.dot(t) + mRawD);
    }

    float EpipolarTriangle::ComputeDistanceToVertex(const cv::Mat &x3Dw, const int &idx)
    {
        if (idx == 0)
            return x3Dw.dot(mX);
        else if (idx == 1)
            return x3Dw.dot(mCl);
        else if (idx == 2)
            return x3Dw.dot(mCr);
    }












}

