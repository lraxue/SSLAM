

#include <Frame.h>
#include <GlobalParameters.h>
#include <Values.h>
#include <Preprocessor.h>
#include <Tracker.h>
#include <FeatureDetector.h>

#include <Monitor.h>
#include <glog/logging.h>

#include <iostream>
#include <include/ETFSLAM.h>

using namespace cv;
using namespace SSLAM;


int main() {

    const std::string settingFile = strSettingFileKitti00_02;
    const std::string imagePath = strImagePathKITTI00;
    std::vector<std::string> vstrImageLeft;
    std::vector<std::string> vstrImageRight;

    // Preprocessor::LoadImagesUisee(imagePath, vstrImageLeft, vstrImageRight);
    Preprocessor::LoadImagesKitti(imagePath, vstrImageLeft, vstrImageRight);
//    GlobalParameters::LoadParameters(ProjectPath + settingFile);
//    GlobalParameters::LoadParameters(ProjectPath + "Stereo/adirondack.yaml"); // strSettingFile

    // Main Loop

    // Tracker tracker;
    ETFSLAM slam(ProjectPath + settingFile);
    int nImages = vstrImageLeft.size();
    LOG(INFO) << "Number of images: " << nImages;

    // sleep(20);
    int startIdx = 0;
    int endIdx = nImages;
    int step = 1;
    for (int i = startIdx; i < endIdx; i += step)
    {
        LOG(INFO) << "Process Frame: " << i << " ......";
        cv::Mat imLeft = imread( vstrImageLeft[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat imRight = imread(vstrImageRight[i], CV_LOAD_IMAGE_GRAYSCALE);

//        cv::Mat imLeft = imread(path + "im0.png", CV_LOAD_IMAGE_GRAYSCALE);
//        cv::Mat imRight = imread(path + "im1.png", CV_LOAD_IMAGE_GRAYSCALE);

        // Frame currentFrame = Frame(imLeft, imRight);


        // tracker.GrabStereo(imLeft, imRight);
        slam.ProcessStereoImage(imLeft, imRight);

        // cv::waitKey(0);

        LOG(INFO) << "Process Frame: " << i << " finished ......";
        cout << endl << endl;
    }

//    slam.SaveTrajectoryRotation("Files/trajectory-kitti-00.txt");
    slam.SaveTrajectoryQuaternion("Results/trajectory-kitti-01.txt");

//    slam.SaveTrajectoryKITTI("Files/trajectory-pku-desk-100.txt");
//    slam.SaveAngleCorrespondedToOneMapPoint("Files/angle-pku-desk-100.txt");

    slam.Shutdown();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}