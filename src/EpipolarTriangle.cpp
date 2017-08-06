//
// Created by feixue on 17-6-21.
//

#include <EpipolarTriangle.h>

#include <Eigen/Eigen>
#include <glog/logging.h>

namespace SSLAM
{
    unsigned long EpipolarTriangle::mnNext = 0;

    EpipolarTriangle::EpipolarTriangle()
    {}

    EpipolarTriangle::EpipolarTriangle(const unsigned long& frameId, const cv::Mat &normal, const float &d) : mNormal(normal), mD(d), mAngle1(0.0), mAngle2(0.0), mAngle3(0.0)
    {
        mnId = mnNext++;

        // Frame id
        mnFrameId = frameId;
    }

    EpipolarTriangle::EpipolarTriangle(const unsigned long& frameId, const cv::Mat& X, const cv::Mat& Cl, const cv::Mat& Cr):mAngle1(0.0), mAngle2(0.0), mAngle3(0.0)
    {
        mnId = mnNext++;

        mnFrameId = frameId;

        // Initialize the three vertexes
        mX = X.clone();
        mCl = Cl.clone();
        mCr = Cr.clone();

        // Normal and distance
        cv::Mat X12 = mCl - mX;
        cv::Mat X13 = mCl - mCr;

        cv::Mat normal = X12.cross(X13);

        mNormal = normal / cv::norm(normal);
        mD = cv::norm(mX.t() * mNormal);

        mRawNormal = mNormal;
        mRawD = mD;

        // Compute three angles
        ComputeThreeAngles();

    }

    EpipolarTriangle::EpipolarTriangle(const unsigned long &frameId, const cv::Mat &X, const cv::Mat &Cl, cv::Mat &Cr,
                                       const SUncertainty &uncertainty)
    {
        mnId = mnNext++;

        mnFrameId = frameId;

        // Initialize the three vertexes
        mX = X.clone();
        mCl = Cl.clone();
        mCr = Cr.clone();

        // Normal and distance
        cv::Mat X12 = mCl - mX;
        cv::Mat X13 = mCl - mCr;

        cv::Mat normal = X12.cross(X13);

        mNormal = normal / cv::norm(normal);
        mD = cv::norm(mX.t() * mNormal);

        mRawNormal = mNormal;
        mRawD = mD;

        // Compute three angles
        ComputeThreeAngles();

        // Set uncertainty
        mUncertainty = uncertainty;
    }

    EpipolarTriangle::~EpipolarTriangle()
    {

    }

    cv::Mat EpipolarTriangle::GetX() const
    {
        return mX.clone();
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

    void EpipolarTriangle::ComputeThreeAngles()
    {
        cv::Mat e12 = mX - mCl;
        cv::Mat e13 = mX - mCr;
        cv::Mat e23 = mCl - mCr;
        double len12 = cv::norm(e12);
        double len13 = cv::norm(e13);
        double len23 = cv::norm(e23);

        double cos1 = (len12 * len12 + len13 * len13 - len23 * len23) / (2 * len12 * len13);
        double cos2 = (len12 * len12 + len23 * len23 - len13 * len13) / (2 * len12 * len23);
        double cos3 = (len13 * len13 + len23 * len23 - len12 * len12) / (2 * len13 * len23);

        mAngle1 = std::acos(cos1) * 180.f / CV_PI;
        mAngle2 = std::acos(cos2) * 180.f / CV_PI;
        mAngle3 = std::acos(cos3) * 180.f / CV_PI;

//        LOG(INFO) << "len12: " << len12 << " ,len13: " << len13 << " ,len23: " << len23;
//        LOG(INFO) << "angle1: " << mAngle1 << " ,angle2: " << mAngle2 << " ,angle3: " << mAngle3  << " ,total: " << mAngle1 + mAngle2 + mAngle3;

    }

    float EpipolarTriangle::Angle1() const
    {
        return mAngle1;
    }

    float EpipolarTriangle::Angle2() const
    {
        return mAngle2;
    }

    float EpipolarTriangle::Angle3() const
    {
        return mAngle3;
    }

}

