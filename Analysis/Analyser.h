//
// Created by feixue on 17-8-12.
//

#ifndef SSLAM_ANALYSER_H
#define SSLAM_ANALYSER_H

#include <Frame.h>
#include <opencv2/core/core.hpp>

namespace SSLAM
{
    struct SRepError
    {

        SRepError(){}
        SRepError(const int& width, const int& height, const float& baseline,
                  const cv::Mat& K, const cv::Mat& T) :
                mnImageWidth(width), mnImageHeight(height), mBaseline(baseline), mK(K.clone()), mT(T.clone())
        {

        }

        void SetImageInfo(const int& width, const int& height,
                          const float& baseline, const cv::Mat& K, const cv::Mat& T)
                {
                    mnImageWidth = width;
                    mnImageHeight = height;
                    mBaseline = baseline;
                    mK = K.clone();
                    mT = T.clone();
                }

        void SetSourceInfo(const cv::Point2f& p, const int& scale, const float& response,
                           const float& match, const float& angle, const float& fuse, const float& depth)
        {
            sPoint = p;
            sScale = scale;
            sUResponse = response;
            sUMatch = match;
            sUAngle = angle;
            sUFuse = fuse;
            sDepth = depth;
        }

        void SetTargetInfo(const cv::Point2f& p, const int& scale, const float& response,
                           const float& match, const float& angle, const float& fuse, const float& depth)
        {
            tPoint = p;
            tScale = scale;
            tUResponse = response;
            tUMatch = match;
            tUAngle = angle;
            tUFuse = fuse;
            tDepth = depth;
        }
        // Information of image
        int mnImageWidth, mnImageHeight;
        cv::Mat mK;
        float mBaseline;
        cv::Mat mT;

        float error;

        // Source information
        cv::Point2f sPoint;
        int sScale;
        float sUResponse;
        float sUMatch;
        float sUAngle;
        float sUFuse;
        float sDepth;

        // Target information
        cv::Point2f tPoint;
        int tScale;
        float tUResponse;
        float tUMatch;
        float tUAngle;
        float tUFuse;
        float tDepth;
    };


    class Analyser
    {
    public:
        Analyser();
        ~Analyser();

    public:

        void Analize(const std::vector<Frame>& vFrames);

        std::vector<SRepError> ComputeReprojectionError(const Frame& lastFrame, const Frame& currentFrame, const float& th = 5);

        void WriteToFile(const std::string& fileName, std::vector<SRepError>& vRepErrors);


    };
}

#endif //SSLAM_ANALYSER_H
