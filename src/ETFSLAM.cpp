//
// Created by feixue on 17-6-28.
//

#include <ETFSLAM.h>
#include <glog/logging.h>

namespace SSLAM
{
    ETFSLAM::ETFSLAM(const std::string &strSettingFile)
    {
        GlobalParameters::LoadParameters(strSettingFile);

        mpMap = new Map();

        mpFrameDrawer = new FrameDrawer(mpMap);
        mpMapDrawer = new MapDrawer(mpMap);
        mpLocalMapper = new LocalMapper(mpMap);

        mpTracker = new Tracker(mpFrameDrawer, mpMapDrawer, mpLocalMapper, mpMap);

        mpViewer = new Viewer(mpFrameDrawer, mpMapDrawer, mpTracker);

        // ptrViewerThread = new thread(&Viewer::Run, mpViewer);
    }

    ETFSLAM::~ETFSLAM()
    {
//        if (ptrViewerThread)
//            delete ptrViewerThread;
//        if (mpMapDrawer)
//            delete mpMapDrawer;
//        if (mpFrameDrawer)
//            delete mpFrameDrawer;
//
//        if (mpViewer)
//            delete mpViewer;
//
//        if (mpMap)
//            delete mpMap;
//
//        if (mpTracker)
//            delete mpTracker;


    }

    void ETFSLAM::ProcessStereoImage(const cv::Mat &imLeft, const cv::Mat &imRight)
    {
        GenerateFrame(imLeft, imRight);
        // mpTracker->GrabStereo(imLeft, imRight);
    }

    void ETFSLAM::Shutdown()
    {
        if (mpViewer)
        {
            mpViewer->RequestFinish();
            while (!mpViewer->isFinish())
            {
                usleep(5000);
            }
        }

        if (mpViewer)
            pangolin::BindToContext("SSLAM: Map Viewer");
    }

    void ETFSLAM::SaveTrajectoryKITTI(const std::string& strTrajectoryFile)
    {
        fstream file(strTrajectoryFile.c_str(), ios::in | ios::out);
        if (!file.is_open())
        {
            LOG(ERROR) << "Open file " << strTrajectoryFile << " error, please check.";
            return;
        }

        LOG(INFO) << "Save trajectory to file " << strTrajectoryFile;

        std::vector<KeyFrame*> allKeyFrames = mpMap->GetAllKeyFrames();
        sort(allKeyFrames.begin(), allKeyFrames.end(), KeyFrame::lId);

        for (auto pKF : allKeyFrames)
        {
            const cv::Mat Tcw = pKF->GetPose();
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            file << setprecision(9) <<
                 Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " " <<
                 Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " " <<
                 Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
        }

        file.close();

        LOG(INFO) << "Trajectory saved!";
    }

    void ETFSLAM::GenerateFrame(const cv::Mat &imLeft, const cv::Mat &imRight)
    {
        mCurrentFrame = Frame(imLeft, imRight);
        mCurrentFrame.SetPose(mvTcws[mCurrentFrame.mnId]);
        mCurrentFrame.GenerateAllMapPoints();

        if (mCurrentFrame.mnId > 0)
        {
            analyser.Analize(mLastFrame, mCurrentFrame);
        }

        mLastFrame = Frame(mCurrentFrame);
    }

    void ETFSLAM::LoadGroundTruthKitti(const std::string &filename)
    {
        mvTcws.clear();

        std::fstream file(filename.c_str(), std::ios::in);
        if (!file.is_open())
        {
            LOG(ERROR) << "Open file " << filename << " error.";
        }

        while (!file.eof())
        {
            std::string line;
            getline(file, line);

            stringstream ss;
            ss << line;

            cv::Mat Rwc = cv::Mat(3, 3, CV_32F);
            cv::Mat twc = cv::Mat(3, 1, CV_32F);

            ss >> Rwc.at<float>(0, 0) >> Rwc.at<float>(0, 1) >> Rwc.at<float>(0, 2) >> twc.at<float>(0)
               >> Rwc.at<float>(1, 0) >> Rwc.at<float>(1, 1) >> Rwc.at<float>(1, 2) >> twc.at<float>(1)
               >> Rwc.at<float>(2, 0) >> Rwc.at<float>(2, 1) >> Rwc.at<float>(2, 2) >> twc.at<float>(2);

            cv::Mat Rcw = Rwc.t();
            cv::Mat tcw = -Rwc.t() * twc;
            cv::Mat Tcw = cv::Mat::ones(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mvTcws.push_back(Tcw);
        }

        file.close();

        LOG(INFO) << "Load groundtruth from file " << filename << " finished.";
    }

    void ETFSLAM::SaveAngleCorrespondedToOneMapPoint(const std::string& strAngleFile)
    {
        fstream file(strAngleFile.c_str(), ios::in | ios::out);
        if (!file.is_open())
        {
            LOG(ERROR) << "Open file " << strAngleFile << " error, please check.";
            return;
        }

        LOG(INFO) << "Save angle to file " << strAngleFile;

        std::vector<MapPoint*> allMapPoints = mpMap->GetAllMapPoints();

        sort(allMapPoints.begin(), allMapPoints.end(), MapPoint::lId);

        // Extract angle
        for (auto pMP : allMapPoints)
        {
            if (!pMP) continue;

            std::vector<EpipolarTriangle*> allETriangles = pMP->GetAllTriangles();
            if (allETriangles.empty()) continue;   // Shouldn't happen

            file << pMP->mnId << " " << allETriangles.size() << endl;     // Id of the MapPoint
            for (auto pET : allETriangles)
            {
                const float angle1 = pET->Angle1();
                const float angle2 = pET->Angle2();
                const float angle3 = pET->Angle3();

                file << setprecision(9) << angle1 << " " << angle2 << " " << angle3 << endl;
            }
        }

        file.close();

        LOG(INFO) << "Save angle finished!";
    }

    void ETFSLAM::SaveObservationsInfo(const std::string &strObservationFile)
    {
        fstream file(strObservationFile.c_str(), ios::in | ios::out);
        if (!file.is_open())
        {
            LOG(ERROR) << "Open file " << strObservationFile << " error!";
            return;
        }

        LOG(INFO) << "Save observation information to file " << strObservationFile;

        std::vector<MapPoint*> allMapPoints = mpMap->GetAllMapPoints();
        sort(allMapPoints.begin(), allMapPoints.end(), MapPoint::lId);

        // Extract information
        for (auto pMP : allMapPoints)
        {
            if (!pMP)
                continue;
            if (pMP->IsBad())
                continue;

            std::map<KeyFrame*, int> obs = pMP->GetAllObservations();

            // Extract observations
            std::vector<std::pair<unsigned long, int> > vAllObs;
            for (auto mit : obs)
            {
                if (!mit.first)
                    continue;
                vAllObs.push_back(std::make_pair(mit.first->mnId, mit.second));
            }

            sort(vAllObs.begin(), vAllObs.end());

            // Record MapPoint id and number of observations
            if (vAllObs.empty())
                continue;

            file << pMP->mnId << " " << vAllObs.size() << endl;
            for (auto Obsi : vAllObs)
            {
                file << Obsi.first << " " << Obsi.second << endl;
            }
        }

        file.close();

        LOG(INFO) << "Save observation finished.";
    }
}
