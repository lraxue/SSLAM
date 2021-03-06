//
// Created by feixue on 17-6-14.
//


#include <Converter.h>

namespace SSLAM
{
    g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvT)
    {
        Eigen::Matrix<double, 3, 3> R;
        R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
                cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
                cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

        Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

        return g2o::SE3Quat(R, t);
    }

    cv::Mat Converter::toCvMat(const g2o::SE3Quat &SE3)
    {
        Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();

        return toCvMat(eigMat);
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
    {
        cv::Mat cvMat(3, 3, CV_32F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cvMat.at<float>(i, j) = m(i, j);

        return cvMat.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m)
    {
        cv::Mat cvMat(4, 4, CV_32F);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                cvMat.at<float>(i, j) = m(i, j);

        return cvMat.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &m)
    {
        cv::Mat cvMat(3, 1, CV_32F);
        for (int i = 0; i < 3; ++i)
            cvMat.at<float>(i) = m(i);

        return cvMat.clone();
    }

    cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t)
    {
        cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cvMat.at<float>(i, j) = R(i, j);

        for (int i = 0; i < 3; ++i)
            cvMat.at<float>(i, 3) = t(i);

        return cvMat.clone();

    }

    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector)
    {
        Eigen::Matrix<double, 3, 1> v;
        v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);
    }


}
