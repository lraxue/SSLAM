//
// Created by feixue on 17-7-6.
//

#include <Test.h>
#include <ImageProcessor.h>
#include <Monitor.h>
#include <ORBmatcher.h>

#include <glog/logging.h>

namespace SSLAM
{
    void Test::TestMatch(const Frame &lastFrame, const Frame &currentFrame)
    {
        const std::vector<cv::KeyPoint>& vKeysll = lastFrame.mvKeysLeft;
        const std::vector<cv::KeyPoint>& vKeyslr = lastFrame.mvKeysRight;

        const std::vector<cv::KeyPoint>& vKeyscl = currentFrame.mvKeysLeft;
        const std::vector<cv::KeyPoint>& vKeyscr = currentFrame.mvKeysRight;

        const std::vector<int>& vMatchesl = lastFrame.mvMatches;
        const std::vector<int>& vMatchesc = currentFrame.mvMatches;


        const cv::Mat& mDescll = lastFrame.mDescriptorsLeft;
        const cv::Mat& mDesccl = currentFrame.mDescriptorsLeft;

        // Test Bruteforce match

        std::vector<int> vBFmatches;
        int nBFmatches = ImageProcessor::BruteForceMatch(mDescll, mDesccl, vBFmatches);
        std::vector<cv::KeyPoint> vBFkeysll;
        std::vector<cv::KeyPoint> vBFkeyslr;
        std::vector<cv::KeyPoint> vBFkeyscl;
        std::vector<cv::KeyPoint> vBFkeyscr;
        for (int i = 0, iend = vBFmatches.size(); i < iend; ++i)
        {
            if (vBFmatches[i] < 0)
                continue;
            vBFkeysll.push_back(vKeysll[i]);
            vBFkeyslr.push_back(vKeyslr[vMatchesl[i]]);

            int matchedIdx = vBFmatches[i];
            vBFkeyscl.push_back(vKeyscl[matchedIdx]);
            vBFkeyscr.push_back(vKeyscr[vMatchesc[matchedIdx]]);
        }

        cv::Mat mBFmatched;
        Monitor::DrawMatchesBetweenTwoStereoFrames(lastFrame.mLeft, lastFrame.mRight,
                                                   currentFrame.mLeft, currentFrame.mRight,
                                                   vBFkeysll, vBFkeyslr, vBFkeyscl, vBFkeyscr,
                                                   mBFmatched);

        LOG(INFO) << "BF matches: " << nBFmatches;

        // Test Ransac match
        std::vector<cv::Point2f> vPointsll;
        std::vector<cv::Point2f> vPointslr;
        std::vector<cv::Point2f> vPointcl;
        std::vector<cv::Point2f> vPointcr;

        for (auto p : vKeysll)
            vPointsll.push_back(p.pt);
        for (auto p : vKeyslr)
            vPointslr.push_back(p.pt);
        for (auto p : vKeyscl)
            vPointcl.push_back(p.pt);
        for (auto p : vKeyscr)
            vPointcr.push_back(p.pt);

        std::vector<int> vRAmatches;
        int nRAmatches = ImageProcessor::MatchWithRansac(vPointsll, mDescll, vPointcl, mDesccl, vRAmatches);
        std::vector<cv::KeyPoint> vRAkeysll;
        std::vector<cv::KeyPoint> vRAkeyslr;
        std::vector<cv::KeyPoint> vRAkeyscl;
        std::vector<cv::KeyPoint> vRAkeyscr;
        for (int i = 0, iend = vRAmatches.size(); i < iend; ++i)
        {
            if (vRAmatches[i] < 0)
                continue;
            vRAkeysll.push_back(vKeysll[i]);
            vRAkeyslr.push_back(vKeyslr[vMatchesl[i]]);

            int matchedIdx = vRAmatches[i];
            vRAkeyscl.push_back(vKeyscl[matchedIdx]);
            vRAkeyscr.push_back(vKeyscr[vMatchesc[matchedIdx]]);
        }

        cv::Mat mRAmatched;
        Monitor::DrawMatchesBetweenTwoStereoFrames(lastFrame.mLeft, lastFrame.mRight,
                                                   currentFrame.mLeft, currentFrame.mRight,
                                                   vRAkeysll, vRAkeyslr, vRAkeyscl, vRAkeyscr,
                                                   mRAmatched);
        LOG(INFO) << "Ransac matches: " << nRAmatches;

        // Test Match based on motion
        std::vector<int> vMotionMatches;
        ORBmatcher matcher;
        int nMotionMatches = matcher.MatchWithMotionPrediction(lastFrame, currentFrame, vMotionMatches, 5);

        std::vector<cv::KeyPoint> vMotionkeysll;
        std::vector<cv::KeyPoint> vMotionkeyslr;
        std::vector<cv::KeyPoint> vMotionkeyscl;
        std::vector<cv::KeyPoint> vMotionkeyscr;
        for (int i = 0, iend = vMotionMatches.size(); i < iend; ++i)
        {
            if (vMotionMatches[i] < 0)
                continue;
            vMotionkeysll.push_back(vKeysll[i]);
            vMotionkeyslr.push_back(vKeyslr[vMatchesl[i]]);

            int matchedIdx = vRAmatches[i];
            vMotionkeyscl.push_back(vKeyscl[matchedIdx]);
            vMotionkeyscr.push_back(vKeyscr[vMatchesc[matchedIdx]]);
        }

        cv::Mat mMotionMatched;
        Monitor::DrawMatchesBetweenTwoStereoFrames(lastFrame.mLeft, lastFrame.mRight,
                                                   currentFrame.mLeft, currentFrame.mRight,
                                                   vMotionkeysll, vMotionkeyslr, vMotionkeyscl, vMotionkeyscr,
                                                   mMotionMatched);

        LOG(INFO) << "Motion matches: " << nMotionMatches;

        // Test angle
        std::vector<int> vAngleMatches;
        int nAngleMatches = matcher.MatchWithMotionPredictionAndAngles(lastFrame, currentFrame, vAngleMatches);

        std::vector<cv::KeyPoint> vAnglekeysll;
        std::vector<cv::KeyPoint> vAnglekeyslr;
        std::vector<cv::KeyPoint> vAnglekeyscl;
        std::vector<cv::KeyPoint> vAnglekeyscr;
        for (int i = 0, iend = vAngleMatches.size(); i < iend; ++i)
        {
            if (vAngleMatches[i] < 0)
                continue;
            vAnglekeysll.push_back(vKeysll[i]);
            vAnglekeyslr.push_back(vKeyslr[vMatchesl[i]]);

            int matchedIdx = vAngleMatches[i];
            vAnglekeyscl.push_back(vKeyscl[matchedIdx]);
            vAnglekeyscr.push_back(vKeyscr[vMatchesc[matchedIdx]]);
        }

        cv::Mat mAngleMatched;
        Monitor::DrawMatchesBetweenTwoStereoFrames(lastFrame.mLeft, lastFrame.mRight,
                                                   currentFrame.mLeft, currentFrame.mRight,
                                                   vAnglekeysll, vAnglekeyslr, vAnglekeyscl, vAnglekeyscr,
                                                   mAngleMatched);

        LOG(INFO) << "Angle matches: " << nAngleMatches;


    }
}
