//
// Created by feixue on 17-7-6.
//

#ifndef SSLAM_TESTMATCHWITHTRIANGLES_H
#define SSLAM_TESTMATCHWITHTRIANGLES_H

#include <Frame.h>
#include <ORBmatcher.h>
#include <EpipolarTriangle.h>

namespace SSLAM
{
    class TestTriangleForMatching
    {
    public:
        // Bruteforce match on single/stereo frame with/without triangle
        int static BruteforceMatch(const Frame& LastFrame, Frame& CurrentFrame, const float& th, bool bStereo = true, bool bUseTriangle = false);

        int static RansacMatch(const Frame& LastFrame, Frame& CurrentFrame, const float& th, bool bStereo = true, bool bUseTriangle = false);

    };
}
#endif //SSLAM_TESTMATCHWITHTRIANGLES_H
