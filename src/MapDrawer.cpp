//
// Created by feixue on 17-6-28.
//

#include <MapDrawer.h>
#include <GlobalParameters.h>
#include <glog/logging.h>

namespace SSLAM
{
    MapDrawer::MapDrawer(Map *pMap): mpMap(pMap)
    {

        mKeyFrameSize = GlobalParameters::mKeyFrameSize;
        mKeyFrameLineWidth = GlobalParameters::mKeyFrameLineWidth;
        mGraphLineWidth = GlobalParameters::mGraphLineWidth;
        mPointSize = GlobalParameters::mPointSize;
        mCameraSize = GlobalParameters::mCameraSize;
        mCameraLineWidth = GlobalParameters::mCameraLineWidth;
    }

    MapDrawer::~MapDrawer()
    {

    }

    void MapDrawer::DrawMapPoints()
    {
        const std::vector<MapPoint*>& vpMPs = mpMap->GetAllMapPoints();

        if (vpMPs.empty())
            return;

        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(1.0, 0.0, 0.0); // RGB

        for (int i = 0, iend = vpMPs.size(); i < iend; ++i)
        {
            // TODO
            cv::Mat pos = vpMPs[i]->GetPos();
            glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
        }

        glEnd();
    }

    void MapDrawer::DrawKeyFrames(const bool bDrawKF)
    {
        const float& w = mKeyFrameSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        const std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

        LOG(INFO) << "Number of KFs: " << vpKFs.size();
        if (bDrawKF)
        {
            for (int i = 0, iend = vpKFs.size(); i < iend; ++i)
            {
                KeyFrame* pKF = vpKFs[i];
                cv::Mat T = pKF->GetPoseInverse();

                cv::Mat Twc = T.t();
                LOG(INFO) << "KFs " << pKF->mnId << " pose: " << T;

                glPushMatrix();

                glMultMatrixf(Twc.ptr<GLfloat>(0));

                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f,0.0f,1.0f);
                glBegin(GL_LINES);
                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();

            }
        }
    }

    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
    {
        const float &w = mCameraSize;
        const float h = w * 0.75;
        const float z = w * 0.6;

        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }

    void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
    {
        std::unique_lock<std::mutex> lock(mMutexCamera);
        mCameraPose = Tcw.clone();
    }

    void MapDrawer::GetOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
    {
        LOG(INFO) << "mCameraPose: " << mCameraPose;

        if (!mCameraPose.empty())
        {
            cv::Mat Rwc(3, 3, CV_32F);
            cv::Mat twc(3, 1, CV_32F);

            {
                std::unique_lock<std::mutex> lock(mMutexCamera);
                Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
                twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
            }

            // M = [R | t]
            M.m[0] = Rwc.at<float>(0, 0);
            M.m[1] = Rwc.at<float>(1, 0);
            M.m[2] = Rwc.at<float>(2, 0);
            M.m[3] = 0.0;

            M.m[4] = Rwc.at<float>(0, 1);
            M.m[5] = Rwc.at<float>(1, 1);
            M.m[6] = Rwc.at<float>(2, 1);
            M.m[7] = 0.0;

            M.m[8] = Rwc.at<float>(0, 2);
            M.m[9] = Rwc.at<float>(1, 2);
            M.m[10] = Rwc.at<float>(2, 2);
            M.m[11] = 0.0;

            M.m[12] = twc.at<float>(0);
            M.m[13] = twc.at<float>(1);
            M.m[14] = twc.at<float>(2);
            M.m[15] = 1.0;
        }
        else
            M.SetIdentity();
    }



}

