//
// Created by feixue on 17-6-14.
//

#include <Optimizer.h>
#include <Converter.h>

#include <ExtendedG2O.h>

#include <Thirdparty/g2o/g2o/core/block_solver.h>
#include <Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h>
#include <Thirdparty/g2o/g2o/solvers/linear_solver_dense.h>
#include <Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h>
#include <Thirdparty/g2o/g2o/core/robust_kernel_impl.h>
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>

#include <glog/logging.h>

#include <mutex>

namespace SSLAM
{
    int Optimizer::OptimizePose(Frame &frame)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialMatchedPoints = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(frame.mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = frame.N;
        std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgeStereo;
        std::vector<int> vnIndicesStereo;
        vpEdgeStereo.reserve(N);
        vnIndicesStereo.reserve(N);

        const float deltaStereo = sqrt(7.815);

        for (int i = 0; i < N; ++i)
        {
            MapPoint* pMP = frame.mvpMapPoints[i];
            if (pMP)
            {
                nInitialMatchedPoints++;

                frame.mvbOutliers[i] = false;

                // Set EDGE
                Eigen::Matrix<double, 3, 1> obs;
                const cv::KeyPoint& kp = frame.mvKeysLeft[i];
                const float& kp_r = frame.mvuRight[i];
                obs << kp.pt.x, kp.pt.y, kp_r;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = frame.mvInvLevelSigma2[kp.octave];
                Eigen::Matrix3d info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = frame.fx;
                e->fy = frame.fy;
                e->cx = frame.cx;
                e->cy = frame.cy;
                e->bf = frame.mbf;
                cv::Mat Xw = pMP->GetPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

//                LOG(INFO) << "NO: " << nInitialMatchedPoints << " ,fx: " << e->fx << " ,fy: "
//                          << e->fy << " ,cx: " << e->cx << " ,cy: " << e->cy << " ,bf: " << e->bf;

                optimizer.addEdge(e);

                vpEdgeStereo.push_back(e);
                vnIndicesStereo.push_back(i);
            }
        }

        if (nInitialMatchedPoints < 3)
            return 0;

        LOG(INFO) << "Number of matched points to be optimized: " << nInitialMatchedPoints;

        // Perform 4 iterations
        const float chiStereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (int it = 0; it < 4; ++it)
        {
            vSE3->setEstimate(Converter::toSE3Quat(frame.mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

            for (size_t i = 0, iend = vpEdgeStereo.size(); i < iend; ++i)
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgeStereo[i];

                const size_t idx = vnIndicesStereo[i];

                if (frame.mvbOutliers[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                if (chi2 > chiStereo[it])
                {
                    frame.mvbOutliers[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    e->setLevel(0);
                    frame.mvbOutliers[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of outliers
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        frame.SetPose(pose);

        return nInitialMatchedPoints - nBad;


    }

    int Optimizer::OptimizePoseWithUncertainty(Frame& frame)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        optimizer.setAlgorithm(solver);

        int nInitilaMatchedPoints = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(frame.mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        // TODO
    }

    int Optimizer::PoseOptimization(Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        std::vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        std::vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);


        {
            // unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutliers[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysLeft[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutliers[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysLeft[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        LOG(INFO) << "NO: " << kpUn.pt.x << " " << kpUn.pt.y << " " << kp_ur << " " << invSigma2;

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }


        if (nInitialCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutliers[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                // chi2() == information * error
                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutliers[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutliers[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutliers[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutliers[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutliers[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, Map *pMap)
    {
        // Local KeyFrames
        std::vector<KeyFrame*> vLocalKFs;
        vLocalKFs.push_back(pKF);
        pKF->mnLocalBAForKF = pKF->mnId;

        std::vector<KeyFrame*> vNeighborKFs = pKF->GetVectorCovisibleKeyFrames();
        for (int i = 0, iend = vNeighborKFs.size(); i < iend; ++i)
        {
            KeyFrame* pKFi = vNeighborKFs[i];
            pKFi->mnLocalBAForKF = pKF->mnId;

            vLocalKFs.push_back(pKFi);
        }

        // Local MapPoints seen in local KeyFrames
        std::vector<MapPoint*> vLocalMPs;
        for (std::vector<KeyFrame*>::iterator vit = vLocalKFs.begin(), vend = vLocalKFs.end(); vit != vend; vit++)
        {
            std::vector<MapPoint*> vpMPs = (*vit)->GetMapPointMatches();

            for (int i = 0, iend = vpMPs.size(); i < iend; ++i)
            {
                MapPoint* pMP = vpMPs[i];

                if (!pMP || pMP->IsBad()) continue;

                if (pMP)
                {
                    if (pMP->mnLocalBAForKF != pKF->mnId)
                    {
                        vLocalMPs.push_back(pMP);
                        pMP->mnLocalBAForKF = pKF->mnId;
                    }
                }
            }
        }


        // Search local fixed KeyFrames
        std::vector<KeyFrame*> vLocalFixedKFs;
        for (std::vector<MapPoint*>::iterator vit = vLocalMPs.begin(), vend = vLocalMPs.end(); vit != vend; vit++)
        {
            std::map<KeyFrame*, int> obs = (*vit)->GetAllObservations();
            for (std::map<KeyFrame*, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if (pKFi->mnLocalBAForKF != pKF->mnId && pKFi->mnLocalBAForFixedKF != pKF->mnId)
                {
                    pKFi->mnLocalBAForFixedKF = pKF->mnId;
                    vLocalFixedKFs.push_back(pKFi);
                }

            }
        }

        LOG(INFO) << "Local KeyFrames: " << vLocalKFs.size() << " ,fixed local KeyFrames: "
                  << vLocalFixedKFs.size() << " , MapPoints: " << vLocalMPs.size()
                  << " to optimized in local bundle adjustment.";


        // Begin to optimize
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        unsigned long maxKFid = 0;

        // Set Local KeyFrames vertices
        for (std::vector<KeyFrame*>::iterator vit = vLocalKFs.begin(), vend = vLocalKFs.end(); vit != vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // Set Local Fixed KeyFrames vertices
        for (std::vector<KeyFrame*>::iterator vit = vLocalFixedKFs.begin(), vend = vLocalFixedKFs.end(); vit != vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);

            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // Set MapPoint vertices
        const int nExperctedSize = (vLocalKFs.size() + vLocalFixedKFs.size()) * vLocalMPs.size();

        std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgeStereo;
        vpEdgeStereo.reserve(nExperctedSize);
        std::vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExperctedSize);
        std::vector<MapPoint*> vpEdgeMapPointStereo;
        vpEdgeMapPointStereo.reserve(nExperctedSize);

        const float thHuberStereo = sqrt(7.815);

        for (std::vector<MapPoint*>::iterator vit = vLocalMPs.begin(), vend = vLocalMPs.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;

            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetPos()));

            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const std::map<KeyFrame*, int> obs = pMP->GetAllObservations();

            // Set edges
            for (std::map<KeyFrame*, int>::const_iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
            {
                KeyFrame* pKFi = mit->first;
                Eigen::Matrix<double, 3, 1> observation;
                const cv::KeyPoint& kp = pKFi->mvKeysLeft[mit->second];
                const float& kp_ur = pKFi->mvuRight[mit->second];
                observation << kp.pt.x, kp.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(observation);

                const float& invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberStereo);

                e->fx = pKFi->fx;
                e->fy = pKFi->fy;
                e->cx = pKFi->cx;
                e->cy = pKFi->cy;
                e->bf = pKFi->mbf;

                optimizer.addEdge(e);
                vpEdgeStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKFi);
                vpEdgeMapPointStereo.push_back(pMP);
            }

        }

        // Begin to optimize
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore = true;
        if (bDoMore)
        {
            // Check inlier observation
            for (int i = 0, iend = vpEdgeStereo.size(); i < iend; ++i)
            {
                g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgeStereo[i];
                MapPoint* pMP = vpEdgeMapPointStereo[i];

                if (pMP->IsBad())
                    continue;

                if (e->chi2() > 7.815 || !e->isDepthPositive())
                    e->setLevel(1);

                e->setRobustKernel(0);
            }

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        std::vector<std::pair<KeyFrame*, MapPoint*> > vToErase;
        vToErase.reserve(vpEdgeStereo.size());

        // Check inliers
        for (int i = 0, iend = vpEdgeStereo.size(); i < iend; ++i)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgeStereo[i];
            MapPoint* pMP = vpEdgeMapPointStereo[i];

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(std::make_pair(pKFi, pMP));
            }

        }

        // Get Map Mutex
        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        // Erase failed observations
        if (!vToErase.empty())
        {
            for (int i = 0, iend = vToErase.size(); i < iend; ++i)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPoint(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }


        // Recover optimized data

        // KeyFrames
        for (std::vector<KeyFrame*>::iterator vit = vLocalKFs.begin(), vend = vLocalKFs.end(); vit != vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKFi->SetPose(Converter::toCvMat(SE3quat));
        }

        // MapPoints
        for (std::vector<MapPoint*>::iterator vit = vLocalMPs.begin(), vend = vLocalMPs.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;

            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));

            pMP->SetPos(Converter::toCvMat(vPoint->estimate()));

            pMP->UpdateNormalAndDepth();
        }


    }


    //**********************************Pose optimization on 3D points**************************//
    int Optimizer::OptimisePoseOn3DPoints(Frame &frame)
    {

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(frame.mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);


        // Set MapPoint vertices
        const int N = frame.N;

        std::vector<EdgeProjectRGBDPoseOnly* > vpEdge3DPoint;
        std::vector<size_t > vnIndexEdge3DPoint;
        vpEdge3DPoint.reserve(N);
        vnIndexEdge3DPoint.reserve(N);

        const float delta3DPoint = sqrt(7.815);

        for (int i = 0; i < N; ++i)
        {
            MapPoint* pMP = frame.mvpMapPoints[i];
            if (pMP)
            {
                nInitialCorrespondences++;

                frame.mvbOutliers[i] = false;

                EpipolarTriangle* pTriangle = frame.mvpTriangles[i];
                const cv::Mat Xc = pTriangle->GetX();

                // Set Edge
                const cv::Mat Xw = pMP->GetPos();  // Global 3D point

                EdgeProjectRGBDPoseOnly* e = new EdgeProjectRGBDPoseOnly(Eigen::Vector3d(Xc.at<float>(0), Xc.at<float>(1), Xc.at<float>(2)));

                e->setId(i + 1);
                e->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(vSE3));
                e->setMeasurement(Eigen::Vector3d(Xw.at<float>(0), Xw.at<float>(1), Xw.at<float>(2)));
                e->setInformation(Eigen::Matrix3d::Identity() * 1e4);

                optimizer.addEdge(e);
                vpEdge3DPoint.push_back(e);
                vnIndexEdge3DPoint.push_back(i);
            }
        }

        if (nInitialCorrespondences < 3)
            return 0;

        LOG(INFO) << "Number of matched points to be optimized: " << nInitialCorrespondences;

        // Perform 4 iterations
        const float chi3DPoint[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (int it = 0; it < 4; ++it)
        {
            vSE3->setEstimate(Converter::toSE3Quat(frame.mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

            for (size_t i = 0, iend = vpEdge3DPoint.size(); i < iend; ++i)
            {
                EdgeProjectRGBDPoseOnly* e = vpEdge3DPoint[i];

                const size_t idx = vnIndexEdge3DPoint[i];

                if (frame.mvbOutliers[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                if (chi2 > chi3DPoint[it])
                {
                    frame.mvbOutliers[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    e->setLevel(0);
                    frame.mvbOutliers[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }


        g2o::SE3Quat SE3quat_recov = vSE3->estimate();
        frame.SetPose(Converter::toCvMat(SE3quat_recov));

        return nInitialCorrespondences - nBad;
    }

    void Optimizer::LocalBundleAdjustmentBasedOn3DPoints(KeyFrame *pKF, Map *pMap)
    {
        // TODO
    }

    int Optimizer::ICP(const std::vector<cv::Point3f> &vPoints1, const std::vector<cv::Point3f> &vPoints2,
                              cv::Mat &R, cv::Mat &t, std::vector<bool> &vInliers)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType* linearSolver =
                new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // vertex
        cv::Mat T = cv::Mat::eye(4, 4, CV_32F);
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
        pose->setId(0);
        pose->setEstimate(Converter::toSE3Quat(T));
        optimizer.addVertex(pose);

        // edges
        int index = 1;
        std::vector<EdgeProjectRGBDPoseOnly*> edges;

        for (int i = 0; i < vPoints1.size(); ++i)
        {
            EdgeProjectRGBDPoseOnly* edge = new EdgeProjectRGBDPoseOnly(
                    Eigen::Vector3d(vPoints2[i].x, vPoints2[i].y, vPoints2[i].z));

            edge->setId(i + 1);
            edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
            edge->setMeasurement(Eigen::Vector3d(vPoints1[i].x, vPoints1[i].y, vPoints1[i].z));
            edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);

            optimizer.addEdge(edge);
            edges.push_back(edge);
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO)<< "optimization costs time: " << time_used.count() << " seconds.";

        g2o::SE3Quat SE3quat_recov = pose->estimate();
        cv::Mat T_recov = Converter::toCvMat(SE3quat_recov);
    }

}

