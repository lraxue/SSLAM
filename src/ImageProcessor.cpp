//
// Created by feixue on 17-7-20.
//

#include <ImageProcessor.h>
#include <cvaux.h>

namespace SSLAM
{
    float ImageProcessor::NCC(const cv::Mat &a, const cv::Mat &b)
    {
        assert(a.channels() == 1 && b.channels() == 1);
        assert(a.rows == b.rows && a.cols == b.cols);

//        cv::Mat aCopy = a.clone();
//        cv::Mat bCopy = b.clone();
//
//        cv::Scalar aVal = cv::mean(aCopy);
//        cv::Scalar bVal = cv::mean(bCopy);
//
//        float aMean = aVal[0];
//        float bMean = bVal[0];
//
//        cv::Mat aNormalized = aCopy - aMean * cv::Mat::ones(a.size(), a.type());
//        cv::Mat bNormalized = bCopy - bMean * cv::Mat::ones(a.size(), a.type());


//        cv::Mat square_a = a.mul(a);
//        cv::Mat square_b = b.mul(b);
//        cv::Mat ab = a.mul(b);

//        cv::Scalar sum_square_a = cv::sum(square_a);
//        cv::Scalar sum_square_b = cv::sum(square_b);
//        cv::Scalar sum_square_ab = cv::sum(ab);
//
//        float sa2 = sum_square_a[0];
//        float sb2 = sum_square_b[0];
//        float sab = sum_square_ab[0];

        double sa2 = a.dot(a);
        double sb2 = b.dot(b);
        double sab = a.dot(b);

        return sab/sqrt(sa2 * sb2);
    }

    int ImageProcessor::BruteForceMatch(const cv::Mat &mDesc1,
                                        const cv::Mat &mDesc2,
                                        std::vector<int> &vMatches)
    {
        int N = mDesc1.rows;
        if (N == 0)
            return 0;
        vMatches.resize(N, -1);

        std::vector<cv::DMatch> matches;

        cv::BruteForceMatcher<cv::HammingLUT> matcher;
        matcher.match(mDesc1, mDesc2, matches);       // query, train,

        for (int i = 0, iend = matches.size(); i < iend; ++i)
        {
            cv::DMatch match = matches[i];
            vMatches[match.queryIdx] = match.trainIdx;
        }

        return matches.size();
    }

    int ImageProcessor::MatchWithRansac(const std::vector<cv::Point2f> &vKeys1, const cv::Mat &mDesc1,
                                        const std::vector<cv::Point2f> &vKeys2, const cv::Mat &mDesc2,
                                        std::vector<int> &vMatches)
    {
        std::vector<int> vBFmatches;
        BruteForceMatch(mDesc1, mDesc2, vBFmatches);

        std::vector<cv::Point2f> vP1, vP2;
        std::vector<int> index;
        for (int i = 0, iend = vBFmatches.size(); i < iend; ++i)
        {
            if (vBFmatches[i] < 0)
                continue;
            vP1.push_back(vKeys1[i]);
            vP2.push_back(vKeys2[vBFmatches[i]]);

            index.push_back(i);
        }

        std::vector<uchar> states;
        cv::Mat F = cv::findFundamentalMat(vP1, vP2, states, cv::FM_RANSAC);

        int inliers = 0;
        for (int i = 0; i < states.size(); ++i)
        {
            if (states[i] == 0)
            {
                vBFmatches[index[i]] = -1;
            }
            else
                inliers++;
        }

        vMatches = vBFmatches;

        return inliers;
    }
}

