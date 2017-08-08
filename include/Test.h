//
// Created by feixue on 17-7-6.
//

#ifndef SSLAM_TESTMATCHWITHTRIANGLES_H
#define SSLAM_TESTMATCHWITHTRIANGLES_H

#include <Frame.h>
#include <ORBmatcher.h>
#include <EpipolarTriangle.h>
#include <Frame.h>

namespace SSLAM
{
    class Test
    {
    public:

        static void TestMatch(const Frame& lastFrame, const Frame& currentFrame);
    };
}

#endif //SSLAM_TESTMATCHWITHTRIANGLES_H
