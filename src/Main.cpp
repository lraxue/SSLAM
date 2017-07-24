

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

    std::vector<string> vstrImageLeft;
    std::vector<string> vstrImageRight;

    Preprocessor::LoadImages(strImagePathPKUDesk + "image_0/", vstrImageLeft, vstrImageRight);
    GlobalParameters::LoadParameters(ProjectPath + strSettingFilePKUDesk);
//    GlobalParameters::LoadParameters(ProjectPath + "Stereo/adirondack.yaml"); // strSettingFile

    // Main Loop

    // Tracker tracker;
    ETFSLAM slam(ProjectPath + strSettingFilePKUDesk);
    int nImages = vstrImageLeft.size();
    LOG(INFO) << "Number of images: " << nImages;

    const std::string path = "/home/feixue/Research/Dataset/Stereo/Adirondack-perfect/";
    int startIdx = 0;
    int endIdx = nImages;
    int step = 1;
    for (int i = startIdx; i < endIdx; i += step)
    {
        LOG(INFO) << "Process Frame: " << i << " ......";
        cv::Mat imLeft = imread(strImagePathPKUDesk + "image_0/" + vstrImageLeft[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat imRight = imread(strImagePathPKUDesk + "image_1/" + vstrImageRight[i], CV_LOAD_IMAGE_GRAYSCALE);

//        cv::Mat imLeft = imread(path + "im0.png", CV_LOAD_IMAGE_GRAYSCALE);
//        cv::Mat imRight = imread(path + "im1.png", CV_LOAD_IMAGE_GRAYSCALE);

        // Frame currentFrame = Frame(imLeft, imRight);


        // tracker.GrabStereo(imLeft, imRight);
        slam.ProcessStereoImage(imLeft, imRight);

        // cv::waitKey(0);

        LOG(INFO) << "Process Frame: " << i << " finished ......";
        cout << endl << endl;
    }

    slam.SaveTrajectoryKITTI("Files/trajectory-pku-desk-100.txt");
    slam.SaveAngleCorrespondedToOneMapPoint("Files/angle-pku-desk-100.txt");

    slam.Shutdown();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}