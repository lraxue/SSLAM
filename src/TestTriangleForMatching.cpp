//
// Created by feixue on 17-7-6.
//

#include <TestTriangleForMatching.h>

namespace SSLAM
{
    int TestTriangleForMatching::BruteforceMatch(const Frame &lastFrame, Frame &currentFrame,
                                                        const float &th, bool bStereo, bool bUseTriangle)
    {
        int nMatches = 0;

        // Matching results
        std::vector<cv::KeyPoint> vLastMatchedKeysLeft;
        std::vector<cv::KeyPoint> vLastMatchedKeysRight;
        std::vector<cv::KeyPoint> vCurrMatchedKeysLeft;
        std::vector<cv::KeyPoint> vCurrMatchedKeysRight;

        const float& fx = Frame::fx;
        const float& fy = Frame::fy;
        const float& cx = Frame::cx;
        const float& cy = Frame::cy;
        const float& bf = Frame::mbf;

        const cv::Mat Rcw = currentFrame.mRcw;
        const cv::Mat tcw = currentFrame.mtcw;


        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = lastFrame.mRcw;
        const cv::Mat tlw = lastFrame.mtcw;

        // LOG(INFO) << Rcw << " " << tcw << " " << Rlw << " " << tlw;

        const cv::Mat tlc = Rlw * twc + tlw;

        const bool bForward = tlc.at<float>(2) > currentFrame.mb;
        const bool bBackward = -tlc.at<float>(2) > currentFrame.mb;

        // Match between stereo Frame
        if (bStereo)
        {
            std::vector<int> vMatchesInLastFrame = lastFrame.mvMatches;
            std::vector<int> vMatchesInCurrentFrame = currentFrame.mvMatches;

            for (int iC = 0; iC < lastFrame.N; iC++) {
                MapPoint *pMP = lastFrame.mvpMapPoints[iC];
                if (vMatchesInLastFrame[iC] < 0) continue;          // Circle match: ll-lr

                if (pMP) {

                    if (!lastFrame.mvbOutliers[iC]) {
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


                        if (u < 0 || u >= currentFrame.mnImgWidth)
                            continue;
                        if (v < 0 || v >= currentFrame.mnImgHeight)
                            continue;

                        int nLastOctave = lastFrame.mvKeysLeft[iC].octave;

                        // Search in a window. Size depends on scale
                        float radius = th * currentFrame.mvScaleFactors[nLastOctave];

                        std::vector<int> vIndices;

                        if (bForward)
                            vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave);
                        else if (bBackward)
                            vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, 0, nLastOctave);
                        else
                            vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave - 1, nLastOctave + 1);

                        if (vIndices.empty())
                            continue;

                        // const cv::Mat dMP = pMP->GetDescriptor();

                        const cv::Mat& dLL = lastFrame.mDescriptorsLeft.row(iC);
                        const cv::Mat& dLR = lastFrame.mDescriptorsRight.row(vMatchesInLastFrame[iC]);

                        int bestDist = 256;
                        int bestIdx = -1;

                        for (std::vector<int>::const_iterator vit = vIndices.begin(), vend = vIndices.end();
                             vit != vend; vit++) {
                            const size_t idx = *vit;

                            if (vMatchesInCurrentFrame[idx] < 0) continue;    // Circle match: cl-cr


                            const cv::Mat& dCL = currentFrame.mDescriptorsLeft.row(idx);
                            const cv::Mat& dCR = currentFrame.mDescriptorsRight.row(vMatchesInCurrentFrame[idx]);

                            int distLL = ORBmatcher::DescriptorDistance(dLL, dCL);
                            int distRR = ORBmatcher::DescriptorDistance(dLR, dCR);

                            if (distLL > ORBmatcher::TH_HIGH || distRR > ORBmatcher::TH_HIGH) continue;   // Circle matches: ll-cl, lr-cr
                            float dist = (distLL + distRR) / 2.0;

                            if (dist < bestDist) {
                                bestDist = dist;
                                bestIdx = idx;
                            }
                        }

                        if (bestDist <= ORBmatcher::TH_HIGH) {

                            // Here, use the epipolar triangle for selecting
                            if (bUseTriangle)
                            {
                                // TODO
                                
                            }
                            currentFrame.mvpMapPoints[bestIdx] = pMP;
                            nMatches++;

                            // For debugging
                            vLastMatchedKeysLeft.push_back(lastFrame.mvKeysLeft[iC]);
                            vLastMatchedKeysRight.push_back(lastFrame.mvKeysRight[vMatchesInLastFrame[iC]]);

                            vCurrMatchedKeysLeft.push_back(currentFrame.mvKeysLeft[bestIdx]);
                            vCurrMatchedKeysRight.push_back(currentFrame.mvKeysRight[vMatchesInCurrentFrame[bestIdx]]);

                        }
                    }
                }
            }
        }
        else  // Use the left image for matching
        {
            for (int iC = 0; iC < lastFrame.N; iC++) {
                MapPoint *pMP = lastFrame.mvpMapPoints[iC];

                if (pMP) {

                    if (!lastFrame.mvbOutliers[iC]) {
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


                        if (u < 0 || u >= currentFrame.mnImgWidth)
                            continue;
                        if (v < 0 || v >= currentFrame.mnImgHeight)
                            continue;

                        int nLastOctave = lastFrame.mvKeysLeft[iC].octave;

                        // Search in a window. Size depends on scale
                        float radius = th * currentFrame.mvScaleFactors[nLastOctave];

                        std::vector<int> vIndices;

                        if (bForward)
                            vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave);
                        else if (bBackward)
                            vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, 0, nLastOctave);
                        else
                            vIndices = currentFrame.SearchFeaturesInGrid(u, v, radius, nLastOctave - 1, nLastOctave + 1);

                        if (vIndices.empty())
                            continue;

                        // const cv::Mat dMP = pMP->GetDescriptor();

                        const cv::Mat& dLL = lastFrame.mDescriptorsLeft.row(iC);

                        int bestDist = 256;
                        int bestIdx = -1;

                        for (std::vector<int>::const_iterator vit = vIndices.begin(), vend = vIndices.end();
                             vit != vend; vit++) {
                            const size_t idx = *vit;


                            const cv::Mat& dCL = currentFrame.mDescriptorsLeft.row(idx);

                            int distLL = ORBmatcher::DescriptorDistance(dLL, dCL);

                            if (distLL > ORBmatcher::TH_HIGH) continue;

                            float dist = distLL;

                            if (dist < bestDist) {
                                bestDist = dist;
                                bestIdx = idx;
                            }
                        }

                        if (bestDist <= ORBmatcher::TH_HIGH) {
                            currentFrame.mvpMapPoints[bestIdx] = pMP;
                            nMatches++;

                            // For debugging
                            vLastMatchedKeysLeft.push_back(lastFrame.mvKeysLeft[iC]);
                            vCurrMatchedKeysLeft.push_back(currentFrame.mvKeysLeft[bestIdx]);
                        }
                    }
                }
            }
        }
        return nMatches;
    }
}
