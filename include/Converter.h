//
// Created by feixue on 17-6-14.
//

#ifndef SSLAM_CONVERTER_H
#define SSLAM_CONVERTER_H

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>
#include <Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h>

namespace SSLAM
{
    class Converter
    {
    public:
        // cv::Mat to g2o SE3Quat
        static g2o::SE3Quat toSE3Quat(const cv::Mat& cvT);

        static cv::Mat toCvMat(const g2o::SE3Quat& SE3);

        static cv::Mat toCvMat(const Eigen::Matrix3d& m);

        static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1>& m);

        static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4>& m);

        static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3>& R, const Eigen::Matrix<double, 3, 1>& t);

        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat& cvPoint3);
    };
}

#endif //SSLAM_CONVERTER_H
