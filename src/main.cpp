

#include <Frame.h>
#include <GlobalParameters.h>
#include <Values.h>
#include <Preprocessor.h>
#include <Tracker.h>

#include <glog/logging.h>

#include <iostream>
#include <include/ETFSLAM.h>

using namespace cv;
using namespace SSLAM;

int main() {

    const std::string settingFile = strSettingFileUiseeL1; // strSettingFileKitti00_02;
    const std::string imagePath = strImagePathUiseeL1; //strImagePathKITTI00;
    std::vector<string> vstrImageLeft;
    std::vector<string> vstrImageRight;

    Preprocessor::LoadImagesKitti( imagePath, vstrImageLeft, vstrImageRight);
//    GlobalParameters::LoadParameters(ProjectPath + settingFile);
//    GlobalParameters::LoadParameters(ProjectPath + "Stereo/adirondack.yaml"); // strSettingFile

    // Main Loop

    // Tracker tracker;
    ETFSLAM slam(ProjectPath + settingFile);
    int nImages = vstrImageLeft.size();
    LOG(INFO) << "Number of images: " << nImages;


    for (int i = 0; i < 10; i += 1)
    {
        LOG(INFO) << "Process Frame: " << i << " ......";
        cv::Mat imLeft = imread(vstrImageLeft[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat imRight = imread(vstrImageRight[i], CV_LOAD_IMAGE_GRAYSCALE);

        // tracker.GrabStereo(imLeft, imRight);
        slam.ProcessStereoImage(imLeft, imRight);

        LOG(INFO) << "Process Frame: " << i << " finished ......";
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}