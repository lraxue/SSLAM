//
// Created by feixue on 17-6-28.
//

#include <ETFSLAM.h>
#include <glog/logging.h>
#include <Converter.h>

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

        ptrViewerThread = new thread(&Viewer::Run, mpViewer);
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
        mpTracker->GrabStereo(imLeft, imRight);
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

    void ETFSLAM::SaveTrajectoryRotation(const std::string &strTrajectoryFile)
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

    void ETFSLAM::SaveTrajectoryQuaternion(const std::string &filename)
    {
        cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        //cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];

            // pKF->SetPose(pKF->GetPose()*Two);

//            if(pKF->IsBad())
//                continue;

            cv::Mat R = pKF->GetRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
              << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        }

        f.close();
        cout << endl << "trajectory saved!" << endl;

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
