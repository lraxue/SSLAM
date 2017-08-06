//
// Created by feixue on 17-6-12.
//

#ifndef SSLAM_VALUES_H
#define SSLAM_VALUES_H

#include <string>

using namespace std;

namespace SSLAM {
    const string ProjectPath = "/home/feixue/Research/Code/SLAM/SSLAM/";
    const string strSettingFilePKUDesk = "Stereo/pku-desk.yaml";
    const string strImagePathPKUDesk = "/home/feixue/Research/Dataset/data/Desk/images/02/";

    // KITTI dataset
    const string strSettingFileKitti00_02 = "Stereo/KITTI00-02.yaml";
    const string strImagePathKITTI00 = "/home/feixue/Research/Dataset/dataset/sequences/00/";

    const string strSettingFileKitti03 = "Stereo/KITTI03.yaml";
    const string strImagePathKITTI03 = "/home/feixue/Research/Dataset/dataset/sequences/03/";

    const string strSettingFileKitti04 = "Stereo/KITTI04-12.yaml";
    const string strImagePathKITTI04 = "/home/feixue/Research/Dataset/dataset/sequences/04/";
}

#endif //SSLAM_VALUES_H
