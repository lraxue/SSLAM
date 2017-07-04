//
// Created by feixue on 17-6-9.
//

#include <Monitor.h>

#include <glog/logging.h>

namespace SSLAM
{
    void Monitor::DrawMatchesBetweenStereoFrame(const cv::Mat mLeft, const cv::Mat &mRight,
                                                const std::vector<cv::KeyPoint> &vPointsLeft,
                                                const std::vector<cv::KeyPoint> &vPointsRight, cv::Mat &out)
    {
        if (mLeft.empty() || mRight.empty())
            LOG(INFO) << "The input image is empty, please check.";

        const int w = mLeft.cols;
        const int h = mLeft.rows;

        out = cv::Mat(h, w * 2, mLeft.type());
        mLeft.copyTo(out.rowRange(0, h).colRange(0, w));
        mRight.copyTo(out.rowRange(0, h).colRange(w, w * 2));

        for(int i = 0, iend = vPointsLeft.size(); i < iend; ++i)
        {
            cv::Point p1 = vPointsLeft[i].pt;
            cv::Point p2 = vPointsRight[i].pt;
            p2.x += w;

            cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);

            cv::line(out, p1, p2, cv::Scalar(0, 255, 0), LINE_THICKNESS);
        }
    }

    void Monitor::RecordFeatureFlow(const cv::Mat &mLeft, const cv::Mat &mRight,
                                    const std::vector<std::vector<cv::KeyPoint> >& vFeatureFlowLeft,
                                    const std::vector<std::vector<cv::KeyPoint> >& vFeatureFlowRight,
                                    cv::Mat &out)
    {
        if (mLeft.empty() || mRight.empty())
            LOG(INFO) << "The input image is empty, please check.";

        const int w = mLeft.cols;
        const int h = mLeft.rows;

        out = cv::Mat(h, w * 2, mLeft.type());
        mLeft.copyTo(out.rowRange(0, h).colRange(0, w));
        mRight.copyTo(out.rowRange(0, h).colRange(w, w * 2));

        cv::Point2f preP1;
        cv::Point2f preP2;

        for (int i = 0, iend = vFeatureFlowLeft.size(); i < iend; ++i)
        {
            bool bFirst = true;
            // Number of points tracked in current feature flow.
            int nP = vFeatureFlowLeft[i].size();

            for (int j = 0; j < nP; ++j)
            {
                const cv::Point2f p1 = vFeatureFlowLeft[i][j].pt;
                cv::Point2f p2 = vFeatureFlowRight[i][j].pt;
                p2.x += w;

                cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
                cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);

                if (bFirst)
                {
                    bFirst = false;
                    preP1 = p1;
                    preP2 = p2;

                    continue;
                }

                cv::line(out, preP1, p1, cv::Scalar(0, 255, 0), LINE_THICKNESS);
                cv::line(out, preP2, p2, cv::Scalar(0, 255, 0), LINE_THICKNESS);
            }
        }
    }


    float Monitor::ComputeDisparityError(const cv::Mat &mD, const std::vector<cv::Point2f> &vLocations,
                                         const std::vector<float> &vDisparities)
    {
        if (mD.empty())
        {
            LOG(INFO) << "Disparity map is empty, please check.";
            return -1.0;
        }

        if (vLocations.size() == 0)
            return 0.0f;

        float totalError = 0.0f;
        for (int i = 0, iend = vLocations.size(); i < iend; ++i)
        {
            const cv::Point2f p = vLocations[i];
            const float realD = mD.at<float>(p.y, p.x);
            totalError += fabsf(realD - vDisparities[i]);
        }

        return totalError / vLocations.size();
    }

    void Monitor::DrawKeyPoints(const cv::Mat &img, const std::vector<cv::Point2f> &points, const cv::Scalar &scalar, cv::Mat& out)
    {
        if (img.empty())
        {
            LOG(INFO) << "Input image is empty, please check.";
            return;
        }

        out = img.clone();

        const int r = 5;
        for (int i = 0, iend = points.size(); i < iend; ++i)
        {
            cv::Point2f pt = points[i];
            cv::Point2f pt1, pt2;
            pt1.x = pt.x - r;
            pt1.y = pt.y - r;
            pt2.x = pt.x + r;
            pt2.y = pt.y + r;

            cv::rectangle(out, pt1, pt2, scalar);
            cv::circle(out, pt, 2, scalar);
        }
    }

    void Monitor::DrawKeyPointsOnStereoFrame(const cv::Mat &mLeft, const cv::Mat &mRight,
                                             const std::vector<cv::KeyPoint> &vKeysLeft,
                                             const std::vector<cv::KeyPoint> &vKeysRight, cv::Mat &out)
    {
        if (mLeft.empty() || mRight.empty())
        {
            LOG(ERROR) << "The input images are empty, please check.";
            return;
        }

        const int w = mLeft.cols;
        const int h = mLeft.rows;

        out = cv::Mat(h, w * 2, mLeft.type());
        mLeft.copyTo(out.rowRange(0, h).colRange(0, w));
        mRight.copyTo(out.rowRange(0, h).colRange(w, w * 2));

        for(int i = 0, iend = vKeysLeft.size(); i < iend; ++i)
        {
            cv::Point p1 = vKeysLeft[i].pt;

            cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);

        }

        for (int i = 0; i < vKeysRight.size(); ++i)
        {
            cv::Point p2 = vKeysRight[i].pt;
            p2.x += w;

            cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
        }

    }

    void Monitor::DrawMatchesWithModifiedPosition(const cv::Mat &mLeft, const cv::Mat &mRight,
                                                  const std::vector<cv::KeyPoint> &vKeysLeft,
                                                  const std::vector<cv::KeyPoint> &vKeysRight,
                                                  const std::vector<cv::KeyPoint> &vKeysRightModified, cv::Mat &out)
    {

        if (mLeft.empty() || mRight.empty())
            LOG(INFO) << "The input image is empty, please check.";

        const int w = mLeft.cols;
        const int h = mLeft.rows;

        out = cv::Mat(h, w * 2, mLeft.type());
        mLeft.copyTo(out.rowRange(0, h).colRange(0, w));
        mRight.copyTo(out.rowRange(0, h).colRange(w, w * 2));

        for(int i = 0, iend = vKeysLeft.size(); i < iend; ++i)
        {
            cv::Point2f p1 = vKeysLeft[i].pt;
            cv::Point2f p2 = vKeysRight[i].pt;
            cv::Point2f p3 = vKeysRightModified[i].pt;
            p2.x += w;
            p3.x += w;

//            if (p1.y < 50 || p1. y > 500)
//                continue;

            cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
//            cv::circle(out, p3, CIRCLE_RADIUS, cv::Scalar(255, 0, 0), CIRCLE_THICKNESS);

            cv::line(out, p1, p2, cv::Scalar(0, 255, 0), LINE_THICKNESS);
        }
    }

    void Monitor::DrawGridOnImage(const cv::Mat &img, const int &gridRows,
                                  const int &gridCols, cv::Mat &out, const cv::Scalar scalar /*= cv::Scalar(0, 0, 255)*/)
    {
        if (img.empty())
        {
            LOG(ERROR) << "The input image is empty, please check.";
            return;
        }

        out = img.clone();
        const int& W = img.cols;
        const int& H = img.rows;

        for (int i = 1; i < gridRows; ++i)
        {
            float offset = i * H / (float)gridRows;
            cv::Point2f p1 = cv::Point2f(0, offset);
            cv::Point2f p2 = cv::Point2f(W - 1, offset);

            cv::line(out, p1, p2, scalar, 2);
        }

        for (int i = 0; i < gridCols; ++i)
        {
            float offset = i * W / (float)gridCols;
            cv::Point2f p1 = cv::Point2f(offset, 0);
            cv::Point2f p2 = cv::Point2f(offset, H - 1);

            cv::line(out, p1, p2, scalar, 2);
        }
    }

    void Monitor::DrawReprojectedPointsOnStereoFrame(const cv::Mat &mLeft, const cv::Mat &mRight,
                                                     const std::vector<cv::KeyPoint> &vKeysLeft,
                                                     const std::vector<cv::KeyPoint> &vKeysRight,
                                                     const std::vector<cv::Point2f> &vProjectedKeysLeft,
                                                     const std::vector<cv::Point2f> &vProjectedKeysRight, cv::Mat &out)
    {
        if (mLeft.empty() || mRight.empty())
        {
            LOG(ERROR) << "The input images are empty, please check.";
            return;
        }

        const int w = mLeft.cols;
        const int h = mLeft.rows;

        out = cv::Mat(h, w * 2, mLeft.type());
        mLeft.copyTo(out.rowRange(0, h).colRange(0, w));
        mRight.copyTo(out.rowRange(0, h).colRange(w, w * 2));

        if (vKeysLeft.empty()) return;

        for (int i = 0, iend = vKeysLeft.size(); i < iend; ++i)
        {
            cv::Point p1 = vKeysLeft[i].pt;
            cv::Point p2 = vKeysRight[i].pt;
            p2.x += w;

            cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);

            cv::line(out, p1, p2, cv::Scalar(0, 255, 0), LINE_THICKNESS);

            cv::Point2f RepP1 = vProjectedKeysLeft[i];
            cv::Point2f RepP2 = vProjectedKeysRight[i];
            RepP2.x += w;

            cv::circle(out, RepP1, CIRCLE_RADIUS, cv::Scalar(255, 0, 0), CIRCLE_THICKNESS);
            cv::circle(out, RepP2, CIRCLE_RADIUS, cv::Scalar(255, 0, 0), CIRCLE_THICKNESS);
        }
    }

    void Monitor::DrawReprojectedPointsOnStereoFrameAfterOptimization(const cv::Mat &mLeft, const cv::Mat &mRight,
                                                                      const std::vector<cv::KeyPoint> &vKeysLeft,
                                                                      const std::vector<cv::KeyPoint> &vKeysRight,
                                                                      const std::vector<cv::Point2f> &vProjectedKeysLeft,
                                                                      const std::vector<cv::Point2f> &vProjectedKeysRight,
                                                                      const std::vector<bool>& vInliers, cv::Mat &out)
    {
        if (mLeft.empty() || mRight.empty())
            LOG(INFO) << "The input image is empty, please check.";

        const int w = mLeft.cols;
        const int h = mLeft.rows;

        out = cv::Mat(h, w * 2, mLeft.type());
        mLeft.copyTo(out.rowRange(0, h).colRange(0, w));
        mRight.copyTo(out.rowRange(0, h).colRange(w, w * 2));

        for(int i = 0, iend = vKeysLeft.size(); i < iend; ++i)
        {
            cv::Point2f p1 = vKeysLeft[i].pt;
            cv::Point2f p2 = vKeysRight[i].pt;
            cv::Point2f p3 = vProjectedKeysLeft[i];
            cv::Point2f p4 = vProjectedKeysRight[i];
            p2.x += w;
            p4.x += w;

//            cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
//            cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
//            cv::line(out, p1, p2, cv::Scalar(0, 255, 0), LINE_THICKNESS);

            if (vInliers[i])
            {
                cv::circle(out, p3, CIRCLE_RADIUS, cv::Scalar(255, 0, 0), CIRCLE_THICKNESS);
                cv::circle(out, p4, CIRCLE_RADIUS, cv::Scalar(255, 0, 0), CIRCLE_THICKNESS);
                cv::line(out, p3, p4, cv::Scalar(0, 255, 0), LINE_THICKNESS);

                continue;
            }

            cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            // cv::line(out, p1, p2, cv::Scalar(0, 255, 0), LINE_THICKNESS);
        }
    }


    void Monitor::DrawMatchesBetweenTwoStereoFrames(const cv::Mat &mLastLeft, const cv::Mat &mLastRight,
                                                    const cv::Mat& mCurrLeft, const cv::Mat& mCurrRight,
                                                    const std::vector<cv::KeyPoint> &vLastKeysLeft,
                                                    const std::vector<cv::KeyPoint> &vLastKeysRight,
                                                    const std::vector<cv::KeyPoint> &vCurrKeysLeft,
                                                    const std::vector<cv::KeyPoint> &vCurrKeysRight, cv::Mat &out)
    {
        if (mLastLeft.empty() || mLastRight.empty()
            || mCurrLeft.empty() || mCurrRight.empty())
        {
            LOG(ERROR) << "The input images are empty, please check.";
            return;
        }

        const int W = mLastLeft.cols;
        const int H = mLastLeft.rows;

        out = cv::Mat(H * 2, W * 2, mLastLeft.type());
        mLastLeft.copyTo(out.rowRange(0, H).colRange(0, W));
        mLastRight.copyTo(out.rowRange(0, H).colRange(W, W * 2));
        mCurrLeft.copyTo(out.rowRange(H, H * 2).colRange(0 ,W));
        mCurrRight.copyTo(out.rowRange(H, H * 2).colRange(W, W * 2));

        for (int i = 0, iend = vLastKeysLeft.size(); i < iend; ++i)
        {
            cv::Point2f p1 = vLastKeysLeft[i].pt;
            cv::Point2f p2 = vLastKeysRight[i].pt;
            cv::Point2f p3 = vCurrKeysLeft[i].pt;
            cv::Point2f p4 = vCurrKeysRight[i].pt;

            p2.x += W;
            p3.y += H;
            p4.x += W;
            p4.y += H;

            cv::circle(out, p1, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            cv::circle(out, p2, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            cv::circle(out, p3, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);
            cv::circle(out, p4, CIRCLE_RADIUS, cv::Scalar(0, 0, 255), CIRCLE_THICKNESS);

            cv::line(out, p1, p2, cv::Scalar(0, 255, 0), LINE_THICKNESS);
            cv::line(out, p3, p4, cv::Scalar(0, 255, 0), LINE_THICKNESS);
            cv::line(out, p1, p3, cv::Scalar(255, 0, 0), LINE_THICKNESS);
            cv::line(out, p2, p4, cv::Scalar(255, 0, 0), LINE_THICKNESS);
        }
    }
}