//
// Created by feixue on 17-6-28.
//

#include <FrameDrawer.h>
#include <GlobalParameters.h>
#include <glog/logging.h>

using namespace std;
namespace SSLAM
{
    FrameDrawer::FrameDrawer(Map *pMap) : mpMap(pMap)
    {

        mnImgWidth = GlobalParameters::mImageWidth;
        mnImgHeight = GlobalParameters::mImageHeight;

        // For stereo image
        mIm = cv::Mat(mnImgHeight, mnImgWidth * 2, CV_8UC3, cv::Scalar(0, 0, 0));

        LOG(INFO) << mnImgHeight << " " << mnImgWidth;
    }

    FrameDrawer::~FrameDrawer()
    {

    }

    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat im;
        std::vector<cv::KeyPoint> vKeysLeft;
        std::vector<cv::KeyPoint> vKeysRight;
        std::vector<cv::Point2f> vProjectedKeysLeft;
        std::vector<cv::Point2f> vProjectedKeysRight;

        std::vector<int> vObservations;

        std::vector<int> vMatches;
        std::vector<bool> vbMap;

        int frameId;
        {
            std::unique_lock<std::mutex> lock(mMutex);

            mIm.copyTo(im);

            vKeysLeft = mvCurrentKeysLeft;
            vKeysRight = mvCurrentKeysRight;

            vProjectedKeysLeft = mvCurrentProjectedKeysLeft;
            vProjectedKeysRight = mvCurrentProjectedKeysRight;

            vObservations = mnObservations;

            vMatches = mvMatches;
            vbMap = mvbMap;

            frameId = mnFrameId;
        }

        if (im.channels() < 3)
            cv::cvtColor(im, im ,CV_GRAY2BGR);

        mnTrackedVO = 0;
        mnTrackedMap = 0;

        const int n = vKeysLeft.size();
        const int radius = 1;

        for (int i = 0; i < n; ++i)
        {
            cv::Point2f pt1, pt2, pt3, pt4;
            pt1 = vKeysLeft[i].pt;
            pt2 = vKeysRight[i].pt;

            pt2.x += mnImgWidth;

            pt3 = vProjectedKeysLeft[i];
            pt4 = vProjectedKeysRight[i];
            pt4.x += mnImgWidth;

            if (vbMap[i])
            {
                cv::circle(im, pt1, radius * vObservations[i], cv::Scalar(255, 0, 0), 2);
                cv::circle(im, pt2, radius * vObservations[i], cv::Scalar(255, 0, 0), 2);
                cv::line(im, pt1, pt2, cv::Scalar(0, 0, 255), 2);

                mnTrackedMap++;
            }
            else
            {
                cv::circle(im, pt1, radius * vObservations[i], cv::Scalar(255, 0, 0), 2);
                cv::circle(im, pt2, radius * vObservations[i], cv::Scalar(255, 0, 0), 2);
                cv::line(im, pt1, pt2, cv::Scalar(255, 0, 0), 2);

                mnTrackedVO++;
            }

//            cv::circle(im, pt3, 5, cv::Scalar(0, 255, 0), 2);
//            cv::circle(im, pt4, 5, cv::Scalar(0, 255, 0), 2);

//            cv::line(im, pt1, pt3, cv::Scalar(0, 255, 0), 2);
//            cv::line(im, pt2, pt4, cv::Scalar(0, 255, 0), 2);
        }

        // cv::imwrite("observations/obs_" + std::to_string(frameId) + ".png", im);

        cv::Mat imWithInfo;
        DrawTextInfo(im, imWithInfo);

//        cv::imshow("mIm", imWithInfo);
//        cv::waitKey(0);

        return imWithInfo;
    }

    void FrameDrawer::DrawTextInfo(cv::Mat &im, cv::Mat &imText)
    {
        std::stringstream ss;

        int nKFs = mpMap->GetKeyFramesInMap();
        int nMPs = mpMap->GetMapPointsInMap();

        ss << "nFs: " << nKFs << ", Global MPs: " << mnTrackedMap << ", Temporal MPs: " << mnTrackedVO << ", Total MPs: " << nMPs;

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
        im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
        imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
        cv::putText(imText, ss.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
    }

    void FrameDrawer::Update(Tracker *pTracker)
    {
        std::unique_lock<std::mutex> lock(mMutex);

        const Frame& frame = pTracker->mCurrentFrame;
        mnFrameId = frame.mnId;
        int nF = frame.N;

        LOG(INFO) << frame.mRGBLeft.size() << " " << frame.mRGBRight.size() << " " << mIm.size() << " " << mnImgHeight << " " << mnImgWidth;


        frame.mRGBLeft.copyTo(mIm.rowRange(0, mnImgHeight).colRange(0, mnImgWidth));
        frame.mRGBRight.copyTo(mIm.rowRange(0, mnImgHeight).colRange(mnImgWidth, mIm.cols));


        const std::vector<int>& vMatches = frame.mvMatches;
        mvCurrentKeysLeft.clear();
        mvCurrentKeysRight.clear();
        mvCurrentProjectedKeysLeft.clear();
        mvCurrentProjectedKeysRight.clear();
        mnObservations.clear();
        mvbMap.clear();

        for (int i = 0; i < nF; ++i)
        {
            MapPoint* pMP = frame.mvpMapPoints[i];
            if (pMP)
            {
                if (pMP->Observations() < 1) continue;
                if (frame.mvbOutliers[i]) continue;

                mvCurrentKeysLeft.push_back(frame.mvKeysLeft[i]);
                mvCurrentKeysRight.push_back(frame.mvKeysRight[vMatches[i]]);

                cv::Point2f pt1 = frame.Project3DPointOnLeftImage(i);
                cv::Point2f pt2 = frame.Project3DPointOnRightImage(i);

                mvCurrentProjectedKeysLeft.push_back(pt1);
                mvCurrentProjectedKeysRight.push_back(pt2);

                mnObservations.push_back(pMP->Observations());

                // TODO
                mvbMap.push_back(true);
            }
        }
    }
}

