//
// Created by feixue on 17-6-12.
//

#ifndef SSLAM_PREPROCESSOR_H
#define SSLAM_PREPROCESSOR_H

#include <vector>
#include <string>

namespace SSLAM
{
    class Preprocessor
    {
    public:

        void static LoadImagesWithTimeStamps(const std::string& strImageSequencePath,
                        std::vector<std::string>& vstrImageLeft,
                        std::vector<std::string>& vstrImageRight,
                        std::vector<double>& vTimeStamps);

        void static LoadImages(const std::string& strImageSequencePath,
                        std::vector<std::string>& vstrImageLeft,
                        std::vector<std::string>& vstrImageRight);
    };
}
#endif //SSLAM_PREPROCESSOR_H
