//
// Created By FeiXue from Peking University.
//
// FeiXue@pku.edu.cn
//
//


#include "ORBmatcher.h"

#include <stdint.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <include/Monitor.h>
#include <glog/logging.h>

using namespace std;
using namespace cv;

namespace SSLAM
{
    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    int ORBmatcher::SearchByProjection(const Frame &lastFrame, Frame &currentFrame, const int th /* =5 */)
    {
        const cv::Mat &currRcw = currentFrame.mRcw;
        const cv::Mat &currtcw = currentFrame.mtcw;
        const float &fx = Frame::fx;
        const float &fy = Frame::fy;
        const float &cx = Frame::cx;
        const float &cy = Frame::cy;
        const float &bf = Frame::mbf;

        // For debugging
        std::vector<cv::KeyPoint> vKeysLeft;
        std::vector<cv::KeyPoint> vKeysRight;
        std::vector<cv::Point2f> vProKeysLeft;
        std::vector<cv::Point2f> vProKeysRight;

        std::vector<cv::KeyPoint> vLastFrameKeysLeft;
        std::vector<cv::KeyPoint> vLastFrameKeysRight;


        // Search global MapPoints existing in last Frame
        const std::vector<int> vMatchesInLastFrame = lastFrame.mvMatches;
        const std::vector<int> vMatchesInCurrentFrame = currentFrame.mvMatches;

        int nMatchedPoints = 0;

        for (int iL = 0; iL < lastFrame.N; ++iL)
        {
            MapPoint *pMP = lastFrame.mvpMapPoints[iL];
            if (!pMP) continue;                     /// DO NOT EXIST
            if (lastFrame.mvbOutliers[iL]) continue; /// NOT ROBUST

            const cv::Mat X3Dc = currRcw * pMP->GetPos() + currtcw;   // From world coordinate to camera coordinate
            const float &z = X3Dc.at<float>(2);
            if (z <= 0) continue;                // Negative depth

            const float &invz = 1.0f / z;
            const float &x = X3Dc.at<float>(0);
            const float &y = X3Dc.at<float>(1);

            const float &pu = x * fx * invz + cx;
            const float &pv = y * fy * invz + cy;

            // LOG(INFO) << "point: " << iL << ", du: " << lastFrame.mvKeysLeft[iL].pt.x - pu << " ,dv: " << lastFrame.mvKeysLeft[iL].pt.y - pv;

            // Must locate in the image plane
            if (pu < 0 || pu >= currentFrame.mnImgWidth || pv < 0 || pv >= currentFrame.mnImgHeight) continue;

            float radius = th * lastFrame.mvKeysLeft[iL].octave;
            std::vector<int> vCandidates = currentFrame.SearchFeaturesInGrid(pu, pv, radius);

            if (vCandidates.empty()) continue;    // No candidate exist

            int nNumberOfCandidates = vCandidates.size();
            int bestIdxR = -1;
            int bestDistance = ORBmatcher::TH_HIGH;
            for (int i = 0; i < nNumberOfCandidates; ++i)
            {
                const int &iC = vCandidates[i];
                if (vMatchesInCurrentFrame[iC] < 0) continue;   // NOT LOOP MATCH

                const cv::KeyPoint &kpLL = lastFrame.mvKeysLeft[iL];
                const cv::KeyPoint &kpLR = lastFrame.mvKeysRight[vMatchesInLastFrame[iL]];
                const cv::Mat &mDesLL = lastFrame.mDescriptorsLeft.row(iL);
                const cv::Mat &mDesLR = lastFrame.mDescriptorsRight.row(vMatchesInLastFrame[iL]);

                // LOG(INFO) << "iL: " << iL << " " << vMatchesInLastFrame[iL] << " ,iC: " << iC << " " << vMatchesInCurrentFrame[iC];

                const cv::KeyPoint &kpCL = currentFrame.mvKeysLeft[iC];
                const cv::KeyPoint &kpCR = currentFrame.mvKeysRight[vMatchesInCurrentFrame[iC]];
                const cv::Mat &mDesCL = currentFrame.mDescriptorsLeft.row(iC);
                const cv::Mat &mDesCR = currentFrame.mDescriptorsRight.row(vMatchesInCurrentFrame[iC]);

                // Pair-Match constraints
                // Scale
                if (abs(kpLL.octave - kpCL.octave) > 1) continue;
                if (abs(kpLR.octave - kpCR.octave) > 1) continue;

                // Distance between descriptors
                const int distLL = ORBmatcher::DescriptorDistance(mDesLL, mDesCL);
                const int distRR = ORBmatcher::DescriptorDistance(mDesLR, mDesCR);

                const int distMean = (distLL + distRR) / 2;

                if (distMean < bestDistance)
                {
                    bestDistance = distMean;
                    bestIdxR = iC;
                }
            }

            if (bestIdxR == -1) continue;
            currentFrame.mvpMapPoints[bestIdxR] = pMP;

            nMatchedPoints++;

            // For debugging
            vKeysLeft.push_back(currentFrame.mvKeysLeft[bestIdxR]);
            vKeysRight.push_back(currentFrame.mvKeysRight[vMatchesInCurrentFrame[bestIdxR]]);
            vProKeysLeft.push_back(cv::Point2f(pu, pv));
            vProKeysRight.push_back(cv::Point2f(pu + bf * invz, pv));

            vLastFrameKeysLeft.push_back(lastFrame.mvKeysLeft[iL]);
            vLastFrameKeysRight.push_back(lastFrame.mvKeysRight[vMatchesInLastFrame[iL]]);
        }

        cv::Mat mProjImg;
        Monitor::DrawReprojectedPointsOnStereoFrame(currentFrame.mRGBLeft,
                                                    currentFrame.mRGBRight,
                                                    vKeysLeft, vKeysRight,
                                                    vProKeysLeft, vProKeysRight,
                                                    mProjImg);
        cv::imshow("Proj image", mProjImg);

        cv::Mat mMatchedImg;
        Monitor::DrawMatchesBetweenTwoStereoFrames(lastFrame.mRGBLeft, lastFrame.mRGBRight,
                                                   currentFrame.mRGBLeft, currentFrame.mRGBRight,
                                                   vLastFrameKeysLeft, vLastFrameKeysRight,
                                                   vKeysLeft, vKeysRight, mMatchedImg);

        cv::namedWindow("matched image", 0);
        cv::imshow("matched image", mMatchedImg);

        cv::waitKey(0);

        return nMatchedPoints;

    }

    int ORBmatcher::SearchCircleMatchesByProjection(const Frame &lastFrame, Frame &currentFrame, const float th)
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const float &fx = Frame::fx;
        const float &fy = Frame::fy;
        const float &cx = Frame::cx;
        const float &cy = Frame::cy;
        const float &bf = Frame::mbf;

        const cv::Mat Rcw = currentFrame.mRcw;
        const cv::Mat tcw = currentFrame.mtcw;


        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = lastFrame.mRcw;
        const cv::Mat tlw = lastFrame.mtcw;

        // LOG(INFO) << Rcw << " " << tcw << " " << Rlw << " " << tlw;

        const cv::Mat tlc = Rlw * twc + tlw;

        const bool bForward = tlc.at<float>(2) > currentFrame.mb;
        const bool bBackward = -tlc.at<float>(2) > currentFrame.mb;

        std::vector<int> vMatchesInLastFrame = lastFrame.mvMatches;
        std::vector<int> vMatchesInCurrentFrame = currentFrame.mvMatches;

        // For debugging
        std::vector<cv::KeyPoint> vKeysLeft;
        std::vector<cv::KeyPoint> vKeysRight;
        std::vector<cv::Point2f> vProKeysLeft;
        std::vector<cv::Point2f> vProKeysRight;

        std::vector<cv::KeyPoint> vLastFrameKeysLeft;
        std::vector<cv::KeyPoint> vLastFrameKeysRight;
        std::vector<int> vIndexInCurrentFrame;

        // *********** End of debugging*************

        for (int iC = 0; iC < lastFrame.N; iC++)
        {
            MapPoint *pMP = lastFrame.mvpMapPoints[iC];
            if (vMatchesInLastFrame[iC] < 0) continue;          // Circle match: ll-lr

            if (pMP)
            {

                if (!lastFrame.mvbOutliers[iC])
                {
                    // Project
                    cv::Mat x3Dw = pMP->GetPos();
                    cv::Mat x3Dc = Rcw * x3Dw + tcw;

                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0 / x3Dc.at<float>(2);

                    if (invzc < 0)
                        continue;

                    float u = fx * xc * invzc + cx;
                    float v = fy * yc * invzc + cy;

//                    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
//                        continue;
//                    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
//                        continue;

                    if (u < 0 || u >= currentFrame.mnImgWidth)
                        continue;
                    if (v < 0 || v >= currentFrame.mnImgHeight)
                        continue;

                    int nLastOctave = lastFrame.mvKeysLeft[iC].octave;

                    // Search in a window. Size depends on scale
                    float radius = th * currentFrame.mvScaleFactors[nLastOctave];

                    vector<int> vIndices;

                    if (bForward)
                        vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave);
                    else if (bBackward)
                        vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, 0, nLastOctave);
                    else
                        vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave - 1, nLastOctave + 1);

                    if (vIndices.empty())
                        continue;

                    // const cv::Mat dMP = pMP->GetDescriptor();

                    const cv::Mat &dLL = lastFrame.mDescriptorsLeft.row(iC);
                    const cv::Mat &dLR = lastFrame.mDescriptorsRight.row(vMatchesInLastFrame[iC]);

                    int bestDist = 256;
                    int bestIdx = -1;

                    for (vector<int>::const_iterator vit = vIndices.begin(), vend = vIndices.end();
                         vit != vend; vit++)
                    {
                        const size_t idx = *vit;

                        if (vMatchesInCurrentFrame[idx] < 0) continue;    // Circle match: cl-cr

//                        LOG(INFO) << "idx: " << idx << " and matches: " << vCurrentMatches[idx];
//                        if (CurrentFrame.mvpMapPoints[idx])
//                            if (CurrentFrame.mvpMapPoints[idx]->Observations() > 0)
//                                continue;

                        const cv::Mat &dCL = currentFrame.mDescriptorsLeft.row(idx);
                        const cv::Mat &dCR = currentFrame.mDescriptorsRight.row(vMatchesInCurrentFrame[idx]);

                        int distLL = DescriptorDistance(dLL, dCL);
                        int distRR = DescriptorDistance(dLR, dCR);

                        if (distLL > TH_HIGH || distRR > TH_HIGH) continue;   // Circle matches: ll-cl, lr-cr
                        float dist = (distLL + distRR) / 2.0;

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx = idx;
                        }
                    }

                    if (bestDist <= TH_HIGH)
                    {
                        currentFrame.mvpMapPoints[bestIdx] = pMP;
                        nmatches++;

                        // For debugging
                        vKeysLeft.push_back(currentFrame.mvKeysLeft[bestIdx]);
                        vKeysRight.push_back(currentFrame.mvKeysRight[vMatchesInCurrentFrame[bestIdx]]);
                        vProKeysLeft.push_back(cv::Point2f(u, v));
                        vProKeysRight.push_back(cv::Point2f(u + bf * invzc, v));

                        vLastFrameKeysLeft.push_back(lastFrame.mvKeysLeft[iC]);
                        vLastFrameKeysRight.push_back(lastFrame.mvKeysRight[vMatchesInLastFrame[iC]]);

                        vIndexInCurrentFrame.push_back(bestIdx);


                        if (mbCheckOrientation)
                        {
                            float rot = lastFrame.mvKeysLeft[iC].angle - currentFrame.mvKeysLeft[bestIdx].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx);
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i != ind1 && i != ind2 && i != ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                    {
                        currentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                        nmatches--;

                        // for debugging
                        const int &idxR = rotHist[i][j];
                        for (int iv = 0, ivend = vIndexInCurrentFrame.size(); iv < ivend; ++iv)
                        {
                            if (vIndexInCurrentFrame[iv] == idxR)
                                vIndexInCurrentFrame[iv] = -1;
                        }

                    }
                }
            }
        }


        LOG(INFO) << "Show matches here.";

//        // For debugging
//        std::vector<cv::KeyPoint> vValidKeysLeft;
//        std::vector<cv::KeyPoint> vValidKeysRight;
//        std::vector<cv::Point2f> vValidProKeysLeft;
//        std::vector<cv::Point2f> vValidProKeysRight;
//
//        std::vector<cv::KeyPoint> vValidLastFrameKeysLeft;
//        std::vector<cv::KeyPoint> vValidLastFrameKeysRight;
//
//        for (int i = 0, iend = vIndexInCurrentFrame.size(); i < iend; ++i)
//        {
//            if (vIndexInCurrentFrame[i] < 0) continue;
//
//            vValidKeysLeft.push_back(vKeysLeft[i]);
//            vValidKeysRight.push_back(vKeysRight[i]);
//            vValidProKeysLeft.push_back(vProKeysLeft[i]);
//            vValidProKeysRight.push_back(vProKeysRight[i]);
//            vValidLastFrameKeysLeft.push_back(vLastFrameKeysLeft[i]);
//            vValidLastFrameKeysRight.push_back(vLastFrameKeysRight[i]);
//        }
//

//        cv::Mat mProjImg;
//        Monitor::DrawReprojectedPointsOnStereoFrame(currentFrame.mRGBLeft,
//                                                    currentFrame.mRGBRight,
//                                                    vValidKeysLeft, vValidKeysRight,
//                                                    vValidProKeysLeft, vValidProKeysRight,
//                                                    mProjImg);
//        cv::imshow("Proj image", mProjImg);
//
//        cv::Mat mMatchedImg;
//        Monitor::DrawMatchesBetweenTwoStereoFrames(lastFrame.mRGBLeft, lastFrame.mRGBRight,
//                                                   currentFrame.mRGBLeft, currentFrame.mRGBRight,
//                                                   vValidLastFrameKeysLeft, vValidLastFrameKeysRight,
//                                                   vValidKeysLeft, vValidKeysRight, mMatchedImg);

//        cv::namedWindow("matched image", 0);
//        cv::imshow("matched image", mMatchedImg);
//
//        const std::string tag = "Frame: " + std::to_string(currentFrame.mnId) + "-";
//        cv::imwrite(tag + "matched image.png", mMatchedImg);
//
//        cv::waitKey(0);

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float &th)
    {
        int nmatches = 0;

        const bool bFactor = th != 1.0;

        for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
        {
            MapPoint *pMP = vpMapPoints[iMP];
            if (!pMP->mbTrackInView)
                continue;

            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            if (bFactor)
                r *= th;

            const vector<int> vIndices =
                    F.SearchFeaturesInGrid(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel],
                                           nPredictedLevel - 1, nPredictedLevel);

            if (vIndices.empty())
                continue;

            const cv::Mat MPdescriptor = pMP->GetDescriptor();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            // Get best and second matches with near keypoints
            for (vector<int>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

//                if (F.mvpMapPoints[idx])
//                    if (F.mvpMapPoints[idx]->Observations() > 0)
//                        continue;

                if (F.mvuRight[idx] > 0)
                {
                    const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
                    if (er > r * F.mvScaleFactors[nPredictedLevel])
                        continue;
                }

                const cv::Mat &d = F.mDescriptorsLeft.row(idx);

                const int dist = DescriptorDistance(MPdescriptor, d);

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeysLeft[idx].octave;
                    bestIdx = idx;
                }
                else if (dist < bestDist2)
                {
                    bestLevel2 = F.mvKeysLeft[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= TH_HIGH)
            {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                F.mvpMapPoints[bestIdx] = pMP;
                nmatches++;
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchMatchesBasedOnEpipolarTriangles(const Frame &lastFrame, Frame &currentFrame, const float th)
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const float &fx = Frame::fx;
        const float &fy = Frame::fy;
        const float &cx = Frame::cx;
        const float &cy = Frame::cy;
        const float &bf = Frame::mbf;

        const cv::Mat Rcw = currentFrame.mRcw;
        const cv::Mat tcw = currentFrame.mtcw;


        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = lastFrame.mRcw;
        const cv::Mat tlw = lastFrame.mtcw;

        // LOG(INFO) << Rcw << " " << tcw << " " << Rlw << " " << tlw;

        const cv::Mat tlc = Rlw * twc + tlw;

        const bool bForward = tlc.at<float>(2) > currentFrame.mb;
        const bool bBackward = -tlc.at<float>(2) > currentFrame.mb;

        std::vector<int> vMatchesInLastFrame = lastFrame.mvMatches;
        std::vector<int> vMatchesInCurrentFrame = currentFrame.mvMatches;

        // For debugging
        std::vector<cv::KeyPoint> vKeysLeft;
        std::vector<cv::KeyPoint> vKeysRight;
        std::vector<cv::Point2f> vProKeysLeft;
        std::vector<cv::Point2f> vProKeysRight;

        std::vector<cv::KeyPoint> vLastFrameKeysLeft;
        std::vector<cv::KeyPoint> vLastFrameKeysRight;
        std::vector<int> vIndexInCurrentFrame;

        // *********** End of debugging*************

        for (int iC = 0; iC < lastFrame.N; iC++)
        {
            MapPoint *pMP = lastFrame.mvpMapPoints[iC];
            if (vMatchesInLastFrame[iC] < 0) continue;          // Circle match: ll-lr

            if (pMP)
            {

                if (!lastFrame.mvbOutliers[iC])
                {
                    // Project
                    cv::Mat x3Dw = pMP->GetPos();
                    cv::Mat x3Dc = Rcw * x3Dw + tcw;

                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0 / x3Dc.at<float>(2);

                    if (invzc < 0)
                        continue;

                    float u = fx * xc * invzc + cx;
                    float v = fy * yc * invzc + cy;

//                    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
//                        continue;
//                    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
//                        continue;

                    if (u < 0 || u >= currentFrame.mnImgWidth)
                        continue;
                    if (v < 0 || v >= currentFrame.mnImgHeight)
                        continue;

                    int nLastOctave = lastFrame.mvKeysLeft[iC].octave;

                    // Search in a window. Size depends on scale
                    float radius = th * currentFrame.mvScaleFactors[nLastOctave];

                    vector<int> vIndices;

                    if (bForward)
                        vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave);
                    else if (bBackward)
                        vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, 0, nLastOctave);
                    else
                        vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave - 1, nLastOctave + 1);

                    if (vIndices.empty())
                        continue;

                    // const cv::Mat dMP = pMP->GetDescriptor();

                    const cv::Mat &dLL = lastFrame.mDescriptorsLeft.row(iC);
                    const cv::Mat &dLR = lastFrame.mDescriptorsRight.row(vMatchesInLastFrame[iC]);

                    const float fLastAngle3 = lastFrame.mvpTriangles[iC]->Angle3();
                    const float fLastAngle2 = lastFrame.mvpTriangles[iC]->Angle2();

                    int bestDist = 256;
                    int bestIdx = -1;

                    for (vector<int>::const_iterator vit = vIndices.begin(), vend = vIndices.end();
                         vit != vend; vit++)
                    {
                        const size_t idx = *vit;

                        if (vMatchesInCurrentFrame[idx] < 0) continue;    // Circle match: cl-cr

//                        LOG(INFO) << "idx: " << idx << " and matches: " << vCurrentMatches[idx];
//                        if (CurrentFrame.mvpMapPoints[idx])
//                            if (CurrentFrame.mvpMapPoints[idx]->Observations() > 0)
//                                continue;


                        const cv::Mat &dCL = currentFrame.mDescriptorsLeft.row(idx);
                        const cv::Mat &dCR = currentFrame.mDescriptorsRight.row(vMatchesInCurrentFrame[idx]);

                        const float fCurrAngle2 = currentFrame.mvpTriangles[idx]->Angle2();
                        const float fCurrAngle3 = currentFrame.mvpTriangles[idx]->Angle3();

                        // TODO
                        const float thAngle = 1.0f;
                        const float dAngle2 = abs(fLastAngle2 - fCurrAngle2);
                        const float dAngle3 = abs(fLastAngle3 - fCurrAngle3);
                        if (dAngle2 > thAngle || dAngle3 > thAngle) continue;    // Angle constraint

                        int distLL = DescriptorDistance(dLL, dCL);
                        int distRR = DescriptorDistance(dLR, dCR);

                        if (distLL > TH_HIGH || distRR > TH_HIGH) continue;   // Circle matches: ll-cl, lr-cr

                        float dist = (distLL + distRR) / 2.0;

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx = idx;
                        }
                    }

                    if (bestDist <= TH_HIGH)
                    {
                        currentFrame.mvpMapPoints[bestIdx] = pMP;
                        nmatches++;

                        // For debugging
                        vKeysLeft.push_back(currentFrame.mvKeysLeft[bestIdx]);
                        vKeysRight.push_back(currentFrame.mvKeysRight[vMatchesInCurrentFrame[bestIdx]]);
                        vProKeysLeft.push_back(cv::Point2f(u, v));
                        vProKeysRight.push_back(cv::Point2f(u + bf * invzc, v));

                        vLastFrameKeysLeft.push_back(lastFrame.mvKeysLeft[iC]);
                        vLastFrameKeysRight.push_back(lastFrame.mvKeysRight[vMatchesInLastFrame[iC]]);

                        vIndexInCurrentFrame.push_back(bestIdx);


                        if (mbCheckOrientation)
                        {
                            float rot = lastFrame.mvKeysLeft[iC].angle - currentFrame.mvKeysLeft[bestIdx].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx);
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i != ind1 && i != ind2 && i != ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                    {
                        currentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                        nmatches--;

                        // for debugging
                        const int &idxR = rotHist[i][j];
                        for (int iv = 0, ivend = vIndexInCurrentFrame.size(); iv < ivend; ++iv)
                        {
                            if (vIndexInCurrentFrame[iv] == idxR)
                                vIndexInCurrentFrame[iv] = -1;
                        }

                    }
                }
            }
        }


        LOG(INFO) << "Show matches here.";

        return nmatches;
    }

    int ORBmatcher::Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float &th)
    {
        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();
        cv::Mat Ow = pKF->GetCameraCenter();

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        int nFused = 0;

        for (auto pMP : vpMapPoints)
        {
            if (!pMP)
                continue;
            if (pMP->IsBad() || pMP->IsInKeyFrame(pKF))
                continue;

            cv::Mat p3Dw = pMP->GetPos();
            cv::Mat p3Dc = Rcw * p3Dw + tcw;

            // Depth must be positive
            if (p3Dc.at<float>(2) < 0.0f)
                continue;

            const float invz = 1 / p3Dc.at<float>(2);
            const float x = p3Dc.at<float>(0) * invz;
            const float y = p3Dc.at<float>(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            if (!pKF->IsInImage(u, v))
                continue;

            const float ur = u - bf * invz;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            cv::Mat PO = p3Dw - Ow;
            const float dist3D = cv::norm(PO);

            // Depth must be inside the scale pyramid of the image
            if (dist3D < minDistance || dist3D > maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            cv::Mat Pn = pMP->GetNormal();

            if (PO.dot(Pn) < 0.5 * dist3D)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
            const std::vector<int> vIndices = pKF->SearchFeaturesInGrid(u, v, radius);

            if (vIndices.empty())
                continue;

            cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;

            for (auto idx : vIndices)
            {
                const cv::KeyPoint &kp = pKF->mvKeysLeft[idx];

                const int &kpLevel = kp.octave;

                if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                    continue;

                if (pKF->mvuRight[idx] >= 0)
                {
                    // Check reprojection error in stereo
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float &kpr = pKF->mvuRight[idx];
                    const float ex = u - kpx;
                    const float ey = v - kpy;
                    const float er = ur - kpr;
                    const float e2 = ex * ex + ey * ey + er * er;

                    if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
                        continue;
                }

                const cv::Mat &dKF = pKF->mDescriptorsLeft.row(idx);

                const int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestIdx < 0)
                continue;

            // If there is already a MapPoint, replace otherwise add new measurement
            if (bestDist <= TH_LOW)
            {
                MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
                if (pMPinKF)
                {
                    if (!pMPinKF->IsBad())
                    {
                        if (pMPinKF->Observations() > pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else
                {
                    pMP->AddObservation(pKF, bestIdx);
                    pKF->AddMapPoint(pMP, bestIdx);
                }

                nFused++;
            }

        }

        return nFused;
    }

    float ORBmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if (viewCos > 0.998)
            return 2.5;
        else
            return 4.0;
    }

    void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++)
        {
            const int s = histo[i].size();
            if (s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float) max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3 < 0.1f * (float) max1)
        {
            ind3 = -1;
        }
    }


    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^*pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }


    /// For debugging
    int ORBmatcher::MatchWithMotionPrediction(const Frame &lastFrame, const Frame &currentFrame, std::vector<int>& vMatches, const float th)
    {
        int nMatchedPoints = 0;

        vMatches.resize(lastFrame.N, -1);

        for (int iL = 0; iL < lastFrame.N; ++iL)
        {
            const cv::KeyPoint& kpll = lastFrame.mvKeysLeft[iL];
            float radius = th * kpll.octave;

            const float& ull = kpll.pt.x;
            const float& vll = kpll.pt.y;
            std::vector<int> vCandidates = currentFrame.SearchFeaturesInGrid(ull, vll, radius);

            if (vCandidates.empty())
                continue;

            int bestIdxR = -1;
            int bestDistance = ORBmatcher::TH_HIGH;
            for (const auto iC : vCandidates)
            {
                const cv::KeyPoint& kpcl = currentFrame.mvKeysLeft[iC];

                const cv::Mat& mDesll = lastFrame.mDescriptorsLeft.row(iL);
                const cv::Mat& mDescl = currentFrame.mDescriptorsLeft.row(iC);

                // Scale
                if (abs(kpll.octave - kpcl.octave) > 1)
                    continue;

                // Distance between descriptors
                const int distll = ORBmatcher::DescriptorDistance(mDesll, mDescl);
                if (distll < bestDistance)
                {
                    bestDistance = distll;
                    bestIdxR = iC;
                }
            }

            if (bestIdxR == -1)
                continue;

            nMatchedPoints++;
            vMatches[iL] = bestIdxR;

        }
        return nMatchedPoints;
    }

    int ORBmatcher::MatchWithMotionPredictionAndAngles(const Frame &lastFrame, const Frame &currentFrame,
                                                       std::vector<int> &vMatches, const float th, const float thAngle)
    {
        int nMatchedPoints = 0;

        vMatches.resize(lastFrame.N, -1);

        const std::vector<int>& vMatchesInLastFrame = lastFrame.mvMatches;
        const std::vector<int>& vMatchesInCurrentFrame = currentFrame.mvMatches;

        for (int iL = 0; iL < lastFrame.N; ++iL)
        {
            // No triangle exists
            if (vMatchesInLastFrame[iL] < 0)
                continue;

            const cv::KeyPoint& kpll = lastFrame.mvKeysLeft[iL];
            const cv::Mat& mDesll = lastFrame.mDescriptorsLeft.row(iL);
            EpipolarTriangle* pETl = lastFrame.mvpTriangles[iL];
            const float fAngle1l = pETl->Angle1();

            float radius = th * kpll.octave;
            const float& ull = kpll.pt.x;
            const float& vll = kpll.pt.y;

            std::vector<int> vCandidates = currentFrame.SearchFeaturesInGrid(ull, vll, radius);
            if (vCandidates.empty())
                continue;

            int bestIdxR = -1;
            int bestDistance = ORBmatcher::TH_HIGH;

            for (const auto iC : vCandidates)
            {
                // No triangle exists
                if (vMatchesInCurrentFrame[iC] < 0)
                    continue;

                const cv::KeyPoint& kpcl = currentFrame.mvKeysLeft[iC];
                const cv::Mat& mDescl = currentFrame.mDescriptorsLeft.row(iC);
                EpipolarTriangle* pETc = currentFrame.mvpTriangles[iC];
                const float fAngle1c = pETc->Angle1();

                // Scale
                if (abs(kpll.octave - kpcl.octave) > 1)
                    continue;

                // Angle
                if (abs(fAngle1l - fAngle1c) < thAngle)
                    continue;

                // Distance between descriptors
                const int distll = ORBmatcher::DescriptorDistance(mDesll, mDescl);
                if (distll < bestDistance)
                {
                    bestDistance = distll;
                    bestIdxR = iC;
                }
            }

            if (bestIdxR == -1)
                continue;


            nMatchedPoints++;
            vMatches[iL] = bestIdxR;
        }

        return nMatchedPoints;
    }
}