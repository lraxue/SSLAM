//
// Created by feixue on 17-7-2.
//

#ifndef SSLAM_EXTENDEDG2O_H
#define SSLAM_EXTENDEDG2O_H

#include <Eigen/Core>

#include <Thirdparty/g2o/g2o/core/base_vertex.h>
#include <Thirdparty/g2o/g2o/core/base_unary_edge.h>
#include <Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h>
#include <Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h>
#include <Thirdparty/g2o/g2o/types/types_sba.h>


// Extended G2O
class EdgeProjectRGBDPoseOnly : public g2o::BaseEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectRGBDPoseOnly(const Eigen::Vector3d& point) : _point(point){}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);

        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map(_point);
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> (_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        // 3x6
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = -z;
        _jacobianOplusXi(0, 2) = y;
        _jacobianOplusXi(0, 3) = -1;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = 0;

        _jacobianOplusXi(1, 0) = z;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(1, 2) = -x;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -1;
        _jacobianOplusXi(1, 5) = 0;

        _jacobianOplusXi(2, 0) = -y;
        _jacobianOplusXi(2, 1) = x;
        _jacobianOplusXi(2, 2) = 0;
        _jacobianOplusXi(2, 3) = 0;
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = -1;
    }

    bool read(istream & in){}
    bool write(ostream& out) const {};

protected:
    Eigen::Vector3d _point;


};

#endif //SSLAM_EXTENDEDG2O_H
