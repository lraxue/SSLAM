

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

    Preprocessor::LoadImages(strImagePathKITTI00 + "image_0/", vstrImageLeft, vstrImageRight);
    GlobalParameters::LoadParameters(ProjectPath + strSettingFileKitti00_02);
//    GlobalParameters::LoadParameters(ProjectPath + "Stereo/adirondack.yaml"); // strSettingFile

    // Main Loop

    // Tracker tracker;
    ETFSLAM slam(ProjectPath + strSettingFileKitti00_02);
    slam.LoadGroundTruthKitti(strGTFileKitti00);

    int nImages = vstrImageLeft.size();
    LOG(INFO) << "Number of images: " << nImages;

    // sleep(20);
    int startIdx = 0;
    int endIdx = nImages;
    int step = 1;
    for (int i = startIdx; i < endIdx; i += step)
    {
        LOG(INFO) << "Process Frame: " << i << " ......";
        cv::Mat imLeft = imread(strImagePathKITTI00 + "image_0/" + vstrImageLeft[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat imRight = imread(strImagePathKITTI00 + "image_1/" + vstrImageRight[i], CV_LOAD_IMAGE_GRAYSCALE);

//        cv::Mat imLeft = imread(path + "im0.png", CV_LOAD_IMAGE_GRAYSCALE);
//        cv::Mat imRight = imread(path + "im1.png", CV_LOAD_IMAGE_GRAYSCALE);

        // Frame currentFrame = Frame(imLeft, imRight);


        // tracker.GrabStereo(imLeft, imRight);
        slam.ProcessStereoImage(imLeft, imRight);

        // cv::waitKey(0);

        LOG(INFO) << "Process Frame: " << i << " finished ......";
        cout << endl << endl;
    }

//    Analyser analyser;
//    analyser.Analize(slam.mvFrames);
//
//    slam.SaveTrajectoryKITTI("Files/trajectory-pku-desk-100.txt");
//    slam.SaveAngleCorrespondedToOneMapPoint("Files/angle-pku-desk-100.txt");

    slam.Shutdown();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}