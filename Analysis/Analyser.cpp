//
// Created by feixue on 17-8-12.
//


#include <Analyser.h>
#include <ORBmatcher.h>
#include <fstream>
#include <glog/logging.h>

namespace SSLAM
{
    Analyser::Analyser()
    {

    }

    Analyser::~Analyser()
    {

    }

    void Analyser::Analize(const std::vector<Frame> &vFrames)
    {
        const std::string filename = "error.txt";
        int nFrames = vFrames.size();

        for (int i = 1; i < nFrames; ++i)
        {
            std::vector<SRepError> vRepErrors = ComputeReprojectionError(vFrames[i - 1], vFrames[i], 15);
            WriteToFile(filename, vRepErrors);

        }
    }

    std::vector<SRepError> Analyser::ComputeReprojectionError(const Frame &lastFrame, const Frame &currentFrame, const float& th)
    {
        // Frame: MapPoint, ETri,

        // First, search matches between two frames
        int nmatches = 0;

        bool mbCheckOrientation = false;

        // Rotation Histogram (to check rotation consistency)
        std::vector<int> rotHist[ORBmatcher::HISTO_LENGTH];
        for (int i = 0; i < ORBmatcher::HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / ORBmatcher::HISTO_LENGTH;

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

        std::vector<std::pair<int, int> > vMatches;   // Matches

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

                    const cv::Mat &dLL = lastFrame.mDescriptorsLeft.row(iC);
                    const cv::Mat &dLR = lastFrame.mDescriptorsRight.row(vMatchesInLastFrame[iC]);

                    int bestDist = 256;
                    int bestIdx = -1;

                    for (std::vector<int>::const_iterator vit = vIndices.begin(), vend = vIndices.end();
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

                        int distLL = ORBmatcher::DescriptorDistance(dLL, dCL);
                        int distRR = ORBmatcher::DescriptorDistance(dLR, dCR);

                        if (distLL > ORBmatcher::TH_HIGH || distRR > ORBmatcher::TH_HIGH) continue;   // Circle matches: ll-cl, lr-cr
                        float dist = (distLL + distRR) / 2.0;

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx = idx;
                        }
                    }

                    if (bestDist <= ORBmatcher::TH_HIGH)
                    {
                        // currentFrame.mvpMapPoints[bestIdx] = pMP;
                        vMatches.push_back(std::make_pair(iC, bestIdx));
                        nmatches++;


                        if (mbCheckOrientation)
                        {
                            float rot = lastFrame.mvKeysLeft[iC].angle - currentFrame.mvKeysLeft[bestIdx].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == ORBmatcher::HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < ORBmatcher::HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx);
                        }
                    }
                }
            }
        }

        ORBmatcher matcher;
        /*
        //Apply rotation consistency
        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            matcher.ComputeThreeMaxima(rotHist, ORBmatcher::HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < ORBmatcher::HISTO_LENGTH; i++)
            {
                if (i != ind1 && i != ind2 && i != ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                    {
                        currentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                        nmatches--;

                    }
                }
            }
        }*/


        // Second, compute reprojection error
        std::vector<SRepError> vRepErrors;

        cv::Mat K = (cv::Mat_<float>(2, 3) << fx, 0, cx, 0, fy, cy);
        cv::Mat R21 = Rcw * Rlw.t();
        cv::Mat t21 = Rcw * (-Rlw.t() * tlw)+ tcw;
        cv::Mat T21 = cv::Mat::eye(4, 4, CV_32F);
        R21.copyTo(T21.rowRange(0, 3).colRange(0, 3));
        t21.copyTo(T21.rowRange(0, 3).col(3));

        const int ImgWidth = lastFrame.mnImgWidth;
        const int ImgHeight = lastFrame.mnImgHeight;

        for (auto match : vMatches)
        {
            const int& idx1 = match.first;
            const int& idx2 = match.second;

            if (idx2 < 0) continue;  // tag for selection

            const cv::Mat x3Dw = lastFrame.mvpMapPoints[idx1]->GetPos();  // 3x1
            const cv::Mat x3Dc = Rcw * x3Dw + tcw;
            const float xc = x3Dc.at<float>(0);
            const float yc = x3Dc.at<float>(1);
            const float invzc = 1.0 / x3Dc.at<float>(2);

            if (invzc < 0)
                continue;

            float u = fx * xc * invzc + cx;
            float v = fy * yc * invzc + cy;

            const cv::Point2f cP = currentFrame.mvKeysLeft[idx2].pt;
            float error = sqrt((u - cP.x) * (u - cP.x) + (v - cP.y) * (v - cP.y));

            // Store in reprojection structure
            SRepError repError(ImgWidth, ImgHeight, lastFrame.mb, K, T21);
            repError.error = error;

            EpipolarTriangle* pET1 = lastFrame.mvpTriangles[idx1];
            repError.SetSourceInfo(lastFrame.mvKeysLeft[idx1].pt,
                                   lastFrame.mvKeysLeft[idx1].octave,
                                   pET1->mResponse,
                                   pET1->mMatchRatio,
                                   pET1->mAngleRatio,
                                   pET1->mFusedUncertainty,
                                   pET1->mDepth);

            EpipolarTriangle* pET2 = currentFrame.mvpTriangles[idx2];
            repError.SetTargetInfo(currentFrame.mvKeysLeft[idx2].pt,
                                   currentFrame.mvKeysLeft[idx2].octave,
                                   pET2->mResponse,
                                   pET2->mMatchRatio,
                                   pET2->mAngleRatio,
                                   pET2->mFusedUncertainty,
                                   pET2->mDepth);

            vRepErrors.push_back(repError);

        }

        return vRepErrors;
    }

    void Analyser::WriteToFile(const std::string& fileName, std::vector<SRepError>& vRepErrors)
    {
        std::fstream file(fileName.c_str());

        if (!file.is_open())
        {
            LOG(ERROR) << "Open file " << fileName << " error.";
        }

        for (auto error : vRepErrors)
        {
            file << std::setprecision(6) << error.mnImageWidth << " " << error.mnImageHeight << " " << error.error << " "
                 << error.sPoint.x << " " << error.sPoint.y << " " << error.sScale << " " << error.sDepth << " " << error.sUResponse << " "
                 << error.sUMatch << " " << error.sUAngle << " " << error.sUFuse << " "
                 << error.tPoint.x << " " << error.tPoint.y << " " << error.tScale << " " << error.tDepth << " " << error.tUResponse << " "
                 << error.tUMatch << " " << error.tUAngle << " " << error.tUFuse << std::endl;
        }

        file.close();
    }
}