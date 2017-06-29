//
// Created by feixue on 17-6-12.
//

#include <Preprocessor.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <dirent.h>
#include <cstring>
#include <algorithm>

using namespace std;
namespace SSLAM
{
    void Preprocessor::LoadImagesWithTimeStamps(const std::string &strPathToSequence, std::vector<std::string> &vstrImageLeft,
                                  std::vector<std::string> &vstrImageRight, std::vector<double> &vTimeStamps)
    {
        ifstream fTimes;
        string strPathTimeFile = strPathToSequence + "/times.txt";
        fTimes.open(strPathTimeFile.c_str());
        while(!fTimes.eof())
        {
            string s;
            getline(fTimes,s);
            if(!s.empty())
            {
                stringstream ss;
                ss << s;
                double t;
                ss >> t;
                vTimeStamps.push_back(t);
            }
        }

        string strPrefixLeft = strPathToSequence + "/image_0/";
        string strPrefixRight = strPathToSequence + "/image_1/";

        const int nTimes = vTimeStamps.size();
        vstrImageLeft.resize(nTimes);
        vstrImageRight.resize(nTimes);

        for(int i=0; i<nTimes; i++)
        {
            stringstream ss;
            ss << setfill('0') << setw(6) << i;
            vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
            vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
        }
    }

    void Preprocessor::LoadImages(const std::string &strImageSequencePath,
                                  std::vector<std::string> &vstrImageLeft, std::vector<std::string> &vstrImageRight)
    {
        DIR* dir;
        struct dirent* ptr;
        if ((dir = opendir(strImageSequencePath.c_str())) == NULL)
        {
            perror("Open dir error...");
            exit(-1);
        }


        while ((ptr=readdir(dir)) != NULL)
        {
            if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
                continue;
            else if (ptr->d_type == 8)
            {
                vstrImageLeft.push_back((string)ptr->d_name);
            }
        }

        closedir(dir);

        std::sort(vstrImageLeft.begin(), vstrImageLeft.end());

        vstrImageRight = vstrImageLeft;
    }
}

