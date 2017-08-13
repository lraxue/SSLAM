//
// Created By FeiXue from Peking University.
//
// FeiXue@pku.edu.cn
//
//

#include "Frame.h"
#include "GlobalParameters.h"
#include "ORBmatcher.h"
#include <FeatureDetector.h>
#include <ImageProcessor.h>

#include <Monitor.h>

#include <glog/logging.h>
#include <memory>
#include <algorithm>
#include <thread>
#include <fstream>


using namespace std;
using namespace cv;
namespace SSLAM
{
	// Static parameters
	float Frame::fx = 0.0f;
	float Frame::fy = 0.0f;
	float Frame::invfx = 0.0f;
	float Frame::invfy = 0.0f;
	float Frame::cx = 0.0f;
	float Frame::cy = 0.0f;
	float Frame::mb = 0.0f;
	float Frame::mbf = 0.0f;
	bool Frame::mbInitialization = false;

	unsigned long Frame::mnNext = 0;
	int Frame::mnImgWidth = 0;
	int Frame::mnImgHeight = 0;

    int Frame::mnGridRows = 0;
    int Frame::mnGridCols = 0;

	float Frame::mAngleTheta = 1.0f;
	float Frame::mMatchTheta = 1.0f;
	float Frame::alphaF = 0.4;
	float Frame::alphaM = 0.2;
	float Frame::alphaA = 0.4;

	Frame::Frame(){}
	Frame::~Frame()
	{
        mpORBextractorLeft.reset();
        mpORBextractorRight.reset();
	}

    Frame::Frame(const Frame &frame) :
            mnId(frame.mnId), N(frame.N), mLeft(frame.mLeft.clone()), mRight(frame.mRight.clone()),
            mvKeysLeft(frame.mvKeysLeft), mvMatches(frame.mvMatches), mvpMapPoints(frame.mvpMapPoints),
			mvpTriangles(frame.mvpTriangles),
            mvKeysRight(frame.mvKeysRight), mvKeysRightWithSubPixel(frame.mvKeysRightWithSubPixel),
            mDescriptorsLeft(frame.mDescriptorsLeft.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
            mvDepth(frame.mvDepth), mvuRight(frame.mvuRight),mvbOutliers(frame.mvbOutliers),
            mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
            mfLogScaleFactor(frame.mfLogScaleFactor), mvScaleFactors(frame.mvScaleFactors),
            mvInvScaleFactors(frame.mvInvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2),
            mvInvLevelSigma2(frame.mvInvLevelSigma2), mvFeaturesInGrid(frame.mvFeaturesInGrid),
            mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
            mRGBLeft(frame.mRGBLeft.clone()), mRGBRight(frame.mRGBRight.clone()), mDisp(frame.mDisp.clone())
    {
        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }

	Frame::Frame(const cv::Mat& imLeft, const cv::Mat& imRight)
	{
        if (!mbInitialization)
        {
            fx = GlobalParameters::fx;
            fy = GlobalParameters::fy;
            invfx = 1.0 / fx;
            invfy = 1.0 / fy;
            cx = GlobalParameters::cx;
            cy = GlobalParameters::cy;
            mb = GlobalParameters::mb;
            mbf = GlobalParameters::mbf;

            mnImgWidth = imLeft.cols;
            mnImgHeight = imLeft.rows;

            mnGridCols = GlobalParameters::mnGridCols;
            mnGridRows = GlobalParameters::mnGridRows;

			mAngleTheta = GlobalParameters::mThAngle;
			mMatchTheta = GlobalParameters::mThMatch;

			alphaF = 0.5;
			alphaM = 0.3;
			alphaA = 0.2;

            mbInitialization = true;
        }

		// Frame Id
		mnId = mnNext++;

        mLeft = imLeft.clone();
        mRight = imRight.clone();
        cv::cvtColor(mLeft, mRGBLeft, CV_GRAY2RGB);
        cv::cvtColor(mRight, mRGBRight, CV_GRAY2RGB);

		mDisp = cv::Mat(mLeft.rows, mLeft.cols, CV_32FC1);

		// Extract Features
		mpORBextractorLeft = std::make_shared<ORBextractor>(GlobalParameters::mnFeatures, GlobalParameters::mfScaleFactor,
			GlobalParameters::mnLevels, GlobalParameters::iniThFAST, GlobalParameters::minThFAST);
		mpORBextractorRight = std::make_shared<ORBextractor>(GlobalParameters::mnFeatures, GlobalParameters::mfScaleFactor,
			GlobalParameters::mnLevels, GlobalParameters::iniThFAST, GlobalParameters::minThFAST);

		std::thread threadLeft(&Frame::ExtractFeatures, this, imLeft, 0);
		std::thread threadRight(&Frame::ExtractFeatures, this, imRight, 1);
        threadLeft.join();
        threadRight.join();

		N = mvKeysLeft.size();
		if (N == 0)
			return;

        LOG(INFO) << "Frame: " << mnId << " extract " << N << " in left image and " << mvKeysRight.size() << " in right image";

        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();


		mvKeysRightWithSubPixel = mvKeysRight;

        // Initialization before processing
		mvbOutliers.resize(N, false);
        mvpMapPoints.resize(N, static_cast<MapPoint*>(NULL));
		mvpTriangles.resize(N, static_cast<EpipolarTriangle*>(NULL));
        mvMatchCosts.resize(N, -1.f);
        mvNCCValues.resize(N, -1.f);
		mvResponse.resize(N, -1.f);
        mvAngleRatios.resize(N, -1.f);
		mvMatchRatios.resize(N, -1.f);
        mvReprojectorErrorLeft.resize(N, -1.f);
        mvReprojectionErrorRight.resize(N, -1.f);
		mvFuseUncertainty.resize(N, -1.f);

		ComputeStereoMatches();

		AssignFeaturesToGrid();

        // ComputeInitialReprojectionError();
        // ComputeNCCValues();  // Compute NCC values of each match


#ifdef USE_TRIANGLE
		GenerateAllEpipolarTriangles();
#endif


		// ComputeUncertainty();  // After generating epipolar triangles

        // Record, for debugging
		// Record();

#ifdef DEBUG_RECORD
        Record();

        RecordKeyPointInfo();

		cv::imwrite("Frame: " + to_string(mnId) + ".png", mDisp);
#endif
	}
	
	void Frame::ExtractFeatures(const cv::Mat& im, const int& tag)
	{
		if (tag == 0)
			mpORBextractorLeft->operator()(im, cv::Mat(), mvKeysLeft, mDescriptorsLeft);
		else
			mpORBextractorRight->operator()(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
	}

	void Frame::ComputeStereoMatches()
	{
		mvuRight = std::vector<float>(N, -1.0f);
		mvDepth = std::vector<float>(N, -1.0f);
        mvMatches = std::vector<int>(N, -1);

		const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

		const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

		//Assign keypoints to row table
		std::vector<std::vector<size_t> > vRowIndices(nRows, std::vector<size_t>());

		for (int i = 0; i < nRows; i++)
			vRowIndices[i].reserve(200);

		const int Nr = mvKeysRight.size();

		for (int iR = 0; iR < Nr; iR++)
		{
			const cv::KeyPoint &kp = mvKeysRight[iR];
			const float &kpY = kp.pt.y;
			const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
			const int maxr = ceil(kpY + r);
			const int minr = floor(kpY - r);

			for (int yi = minr; yi <= maxr; yi++)
				vRowIndices[yi].push_back(iR);
		}

		// Set limits for search
		const float minZ = mb;
		const float minD = 0;
		const float maxD = mbf / minZ;

		// For each left keypoint search a match in the right image
		std::vector<std::pair<int, int> > vDistIdx;
		vDistIdx.reserve(N);

		for (int iL = 0; iL < N; iL++)
		{
			const cv::KeyPoint &kpL = mvKeysLeft[iL];
			const int &levelL = kpL.octave;
			const float &vL = kpL.pt.y;
			const float &uL = kpL.pt.x;

			const std::vector<size_t> &vCandidates = vRowIndices[vL];

			if (vCandidates.empty())
				continue;

            // LOG(INFO) << "Matched points: " << iL << " with candidates " << vCandidates.size();

			const float minU = uL - maxD;
			const float maxU = uL - minD;

			if (maxU < 0)
				continue;

			int bestDist = ORBmatcher::TH_HIGH;
			size_t bestIdxR = 0;

			const cv::Mat &dL = mDescriptorsLeft.row(iL);

			// Compare descriptor to right keypoints
			for (size_t iC = 0; iC < vCandidates.size(); iC++)
			{
				const size_t iR = vCandidates[iC];
				const cv::KeyPoint &kpR = mvKeysRight[iR];

				if (kpR.octave<levelL - 1 || kpR.octave>levelL + 1)
					continue;

				const float &uR = kpR.pt.x;

				if (uR >= minU && uR <= maxU)
				{
					const cv::Mat &dR = mDescriptorsRight.row(iR);
					const int dist = ORBmatcher::DescriptorDistance(dL, dR);

					if (dist < bestDist)
					{
						bestDist = dist;
						bestIdxR = iR;
					}
				}
			}

            // LOG(INFO) << "Matched points: " << iL << " with best candidates " << bestIdxR;

			// Sub-pixel match by correlation
			if (bestDist < thOrbDist)
			{
				// coordinates in image pyramid at keypoint scale
				const float uR0 = mvKeysRight[bestIdxR].pt.x;
				const float scaleFactor = mvInvScaleFactors[kpL.octave];
				const float scaleduL = round(kpL.pt.x*scaleFactor);
				const float scaledvL = round(kpL.pt.y*scaleFactor);
				const float scaleduR0 = round(uR0*scaleFactor);

				// sliding window search
				const int w = 5;
				cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
				IL.convertTo(IL, CV_32F);
				IL = IL - IL.at<float>(w, w) *cv::Mat::ones(IL.rows, IL.cols, CV_32F);
                // IL = IL / 255.f;

				int bestDist = INT_MAX;
				int bestincR = 0;
				const int L = 5;
				std::vector<float> vDists;
				vDists.resize(2 * L + 1);

				const float iniu = scaleduR0 + L - w;
				const float endu = scaleduR0 + L + w + 1;
				if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
					continue;

				for (int incR = -L; incR <= +L; incR++)
				{
					cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
					IR.convertTo(IR, CV_32F);
                    // IR = IR / 255.f;

					IR = IR - IR.at<float>(w, w) *cv::Mat::ones(IR.rows, IR.cols, CV_32F);

					float dist = cv::norm(IL, IR, cv::NORM_L1);
					if (dist < bestDist)
					{
						bestDist = dist;
						bestincR = incR;
					}

					vDists[L + incR] = dist;
				}

				if (bestincR == -L || bestincR == L)
					continue;

				// Sub-pixel match (Parabola fitting)
				const float dist1 = vDists[L + bestincR - 1];
				const float dist2 = vDists[L + bestincR];
				const float dist3 = vDists[L + bestincR + 1];

				const float deltaR = (dist1 - dist3) / (2.0f*(dist1 + dist3 - 2.0f*dist2));

				if (deltaR<-1 || deltaR>1)
					continue;

				// Re-scaled coordinate
				float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

				float disparity = (uL - bestuR);

				if (disparity >= minD && disparity < maxD)
				{
					if (disparity <= 0)
					{
						disparity = 0.01;
						bestuR = uL - 0.01;
					}
					mvDepth[iL] = mbf / disparity;
					mvuRight[iL] = bestuR;
                    mvMatches[iL] = (int)bestIdxR;
					vDistIdx.push_back(std::pair<int, int>(bestDist, iL));

                    // LOG(INFO) << "Matched points are: " << iL << "-" << bestIdxR;

					// Record corrected position based on sub-pixel
					mvKeysRightWithSubPixel[bestIdxR].pt.x = bestuR;
                    mvMatchCosts[iL] = bestDist;
				}
			}
		}

		sort(vDistIdx.begin(), vDistIdx.end());
		const float median = vDistIdx[vDistIdx.size() / 2].first;
		const float thDist = 1.5f*1.4f*median;

		for (int i = vDistIdx.size() - 1; i >= 0; i--)
		{
			if (vDistIdx[i].first < thDist)
				break;
			else
			{
				mvuRight[vDistIdx[i].second] = -1;
				mvDepth[vDistIdx[i].second] = -1;
                mvMatches[vDistIdx[i].second] = -1;
			}
		}
	}

    void Frame::ComputeNCCValues()
    {
        const int w = 2;
        for (int i = 0; i < N; ++i)
        {
            if (mvMatches[i] < 0)
                continue;

            const cv::Point2f p1 = mvKeysLeft[i].pt;
            const cv::Point2f p2 = mvKeysRight[mvMatches[i]].pt;

            const cv::Mat& a = mLeft.rowRange(p1.y - w, p1.y + w).colRange(p1.x - w, p1.x + w);
            const cv::Mat& b = mRight.rowRange(p2.y - w, p2.y + w).colRange(p2.x - w, p2.x + w);

            float ncc = ImageProcessor::NCC(a, b);

            mvNCCValues[i] = ncc;

            LOG(INFO) << "Match: " << i << " with ncc " << ncc;
        }
    }

    void Frame::ComputeInitialReprojectionError()
    {
        for (int i = 0; i < N; ++i)
        {
            if (mvMatches[i] < 0)
                continue;

            const float& vl = mvKeysLeft[i].pt.y;
            const float& vr = mvKeysRight[mvMatches[i]].pt.y;
            mvReprojectionErrorRight[i] = abs(vl - vr) + 0.01;

//            LOG(INFO) << "Key: " << i << " with reprojection error: " << abs(vl - vr) + 0.1;
        }
    }

	void Frame::ComputeUncertainty()
	{

		fstream file("uncertainty_full.txt", ios::in | ios::out);
		if (file.is_open())
		{
			LOG(ERROR) << "OPen file error.";
		}
		for (int i = 0; i < N; ++i)
		{
			if (mvMatches[i] < 0)
				continue;

			// Feature uncertainty based on corner response
			const float rl = mvKeysLeft[i].response / 255.f;
			const float rr = mvKeysRight[i].response / 255.f;
			mvResponse[i] = (rl + rr) / 2.f;

			// Match uncertainty based on reprojection error
			mvMatchRatios[i] = mvReprojectionErrorRight[i] / mMatchTheta;

			// Observation uncertainty based on vertex angle
			mvAngleRatios[i] = mvpTriangles[i]->Angle1() / mAngleTheta;

			mvFuseUncertainty[i] = std::exp(alphaF* mvResponse[i] + alphaA * mvAngleRatios[i] - alphaM * mvMatchRatios[i]);


			file << i << " " << mvResponse[i] << " " << mvMatchRatios[i] << " " << mvAngleRatios[i] << " " << mvFuseUncertainty[i] << endl;
//			LOG(INFO) << "Key: " << i << " Response: " << mvResponse[i] << " Match: " << mvMatchRatios[i] << " Angle: " << mvAngleRatios[i] << " Fuse: " << mvFuseUncertainty[i];
		}

		file.close();
	}

	void Frame::SetPose(const cv::Mat& Pos)
	{
		mTcw = Pos.clone();

		UpdatePoseMatrix();
	}

    cv::Mat Frame::GetPose() const
    {
        return mTcw.clone();
    }

    cv::Mat Frame::GetRotationInverse() const
    {
        return mRwc.clone();
    }

    cv::Mat Frame::GetCameraCenter() const
    {
        return mOw.clone();
    }

	void Frame::UpdatePoseMatrix()
	{
		mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
		mRwc = mRcw.t();
		mtcw = mTcw.rowRange(0, 3).col(3);
		mOw = -mRwc * mtcw;
	}

	cv::Mat Frame::UnprojectStereo(const int& idx)
	{
		if (idx < 0 || idx >= N)
			return cv::Mat();

		const float z = mvDepth[idx];
		if (z > 0)
		{
			const float u = mvKeysLeft[idx].pt.x;
			const float v = mvKeysLeft[idx].pt.y;

			const float x = (u - cx) * z * invfx;
			const float y = (v - cy) * z * invfy;

			cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

			return mRwc * x3Dc + mOw;
		}
		else
			return cv::Mat();
	}

    void Frame::GenerateAllMapPoints()
    {
        for (int i = 0; i < N; ++i)
        {
            if (mvMatches[i] < 0) continue;

            MapPoint* pNewMP = new MapPoint(nullptr, *this, i);
            cv::Mat x3Dw = UnprojectStereo(i);
            pNewMP->SetPos(x3Dw);

            mvpMapPoints[i] = pNewMP;
        }
    }

	EpipolarTriangle* Frame::GenerateEpipolarTriangle(const int &idx) const
    {
        if (idx < 0 || idx >= N)   // out of range
            return static_cast<EpipolarTriangle*>(NULL);

        // Uncertainty
        const float& ResponseLeft = mvKeysLeft[idx].response / 255.f;
        const float& ResponseRight = mvKeysRight[idx].response / 255.f;
        const float& MatchCost = mvMatchCosts[idx];

        // Vertexes
        cv::Mat Cl = (cv::Mat_<float>(3, 1) << 0.f, 0.f, 0.f);
        cv::Mat Cr = (cv::Mat_<float>(3, 1) << mb, 0.f, 0.f);

        const float z = mvDepth[idx];
        const float u = mvKeysLeft[idx].pt.x;
        const float v = mvKeysLeft[idx].pt.y;

        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;

        cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

//        EpipolarTriangle* pNewET =
//                new EpipolarTriangle(mnId, x3Dc, Cl, Cr, sUncertainty);   // EpipolarTriangle in current camera coordinate

        EpipolarTriangle* pNewET =
                new EpipolarTriangle(mnId, x3Dc, Cl, Cr,
                mvKeysLeft[idx], mvKeysRight[mvMatches[idx]],
                mvDepth[idx], mvuRight[idx]);
        return pNewET;
    }

	int Frame::GenerateAllEpipolarTriangles()
	{
		int nValidET = 0;

		for (int i = 0; i < N; ++i)
		{
			EpipolarTriangle* pET = GenerateEpipolarTriangle(i);

			if (pET)
			{
				nValidET++;

				mvpTriangles[i] = pET;
			}
		}
		return nValidET;
	}



    cv::Point2f Frame::Project3DPointOnLeftImage(const int &idx) const
    {
        if (idx < 0 || idx >= N) return cv::Point2f();   // out of range
        if (!mvpMapPoints[idx]) return cv::Point2f();    // Not exist

        const cv::Mat X3Dw = mvpMapPoints[idx]->GetPos();  // Global pose. Embarrassed!
        const cv::Mat X3Dc = mRcw * X3Dw + mtcw;           // Camera coordinate. Excited!
        const float invz = 1.0 / X3Dc.at<float>(2);
        const float x = X3Dc.at<float>(0);
        const float y = X3Dc.at<float>(1);

        const float u = x * fx * invz + cx;
        const float v = y * fy  * invz + cy;

        return cv::Point2f(u, v);
    }

    cv::Point2f Frame::Project3DPointOnRightImage(const int &idx) const
    {
        if (idx < 0 || idx >= N) return cv::Point2f();   // out of range
        if (!mvpMapPoints[idx]) return cv::Point2f();   // Not exist

        const cv::Mat X3Dw = mvpMapPoints[idx]->GetPos();  // Global pose. Embarrassed!
        const cv::Mat X3Dc = mRcw * X3Dw + mtcw;           // Camera coordinate. Excited!
        const float invz = 1.0 / X3Dc.at<float>(2);
        const float x = X3Dc.at<float>(0);
        const float y = X3Dc.at<float>(1);

        const float u = x * fx * invz + cx;
        const float v = y * fy  * invz + cy;

        const float disp = mbf * invz;
        return cv::Point2f(u - disp, v);
    }

    float Frame::ComputeReprojectionError()
    {
        float error = 0.0f;
        int nInliers = 0;

        for (int i = 0; i < N; ++i)
        {
            if (!mvpMapPoints[i]) continue;
            if (mvbOutliers[i]) continue;

            const cv::Mat X3D = mvpMapPoints[i]->GetPos();

            cv::Point2f rp = Project3DPointOnLeftImage(i);
            const cv::Point& p = mvKeysLeft[i].pt;

            const float du = p.x - rp.x;
            const float dv = p.y - rp.y;
            const float dru = mvuRight[i] - (rp.x - mbf / X3D.at<float>(2));
            error += sqrt(du * du + dv * dv + dru * dru);

            nInliers++;
        }

        return error / nInliers;
    }

	void Frame::AssignFeaturesToGrid()
	{
        mvFeaturesInGrid.resize(mnGridRows);
        for (int i = 0; i < mnGridRows; ++i)
            mvFeaturesInGrid[i].resize(mnGridCols);

        // Assign features to feature grid
        for (int i = 0; i < N; ++i)
        {
            const cv::Point2f& pt = mvKeysLeft[i].pt;
            int x = -1, y = -1;
            if (PoseInGrid(pt.x, pt.y, x ,y)) {
                mvFeaturesInGrid[y][x].push_back(i);
                // LOG(INFO) << "Feature " << i << " with position " << pt.y << " " << pt.x << " in grid " << y << " " << x;
            }
        }
	}

    bool Frame::PoseInGrid(const float &u, const float &v, int &x, int &y)
    {
        if (u < 0 || u >= mnImgWidth) return false;
        if (v < 0 || v >= mnImgHeight) return false;

        x = floor(u * mnGridCols / (float)mnImgWidth);
        y = floor(v * mnGridRows / (float)mnImgHeight);

        if (x >= mnImgWidth || y >= mnImgHeight) return false;
        else return true;
    }

	bool Frame::IsInFrustum(MapPoint *pMP, float viewingCosLimit)
	{
		pMP->mbTrackInView = false;

		// 3D position in world coordinate
		cv::Mat P = pMP->GetPos();

		// 3D in camera coordinate
		const cv::Mat Pc = mRcw * P + mtcw;
		const float& PcX = Pc.at<float>(0);
		const float& PcY = Pc.at<float>(1);
		const float& PcZ = Pc.at<float>(2);

		if (PcZ < 0.0f)
			return false;

		// Project in image
		const float invz = 1.0f / PcZ;
		const float u = fx * PcX * invz + cx;
		const float v = fy * PcY * invz + cy;

		if (u < 0 || u >= mnImgWidth)
			return false;
		if (v < 0 || v >= mnImgHeight)
			return false;

		// Check distance is in the scale invariance region of the MapPoint
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const cv::Mat PO = P-mOw;
		const float dist = cv::norm(PO);

		if(dist<minDistance || dist>maxDistance)
			return false;

		// Check viewing angle
		cv::Mat Pn = pMP->GetNormal();

		const float viewCos = PO.dot(Pn)/dist;

		if(viewCos<viewingCosLimit)
			return false;

		// Predict scale in the image
		const int nPredictedLevel = pMP->PredictScale(dist,this);

		// Data used by the tracking
		pMP->mbTrackInView = true;
		pMP->mTrackProjX = u;
		pMP->mTrackProjXR = u - mbf*invz;
		pMP->mTrackProjY = v;
		pMP->mnTrackScaleLevel= nPredictedLevel;
		pMP->mTrackViewCos = viewCos;

		return true;
	}


	std::vector<int> Frame::SearchFeaturesInGrid(const float &cX, const float &cY, const float &radius,
												 const int minLevel, const int maxLevel) const
	{
		std::vector<int> vCandidates;
		vCandidates.reserve(N);

		const float fGridRowPerPixel = mnGridRows / (float)mnImgHeight;
		const float fGridColPerPixel = mnGridCols / (float)mnImgWidth;

		int minX = std::max(0.f, std::floor((cX - radius)*fGridColPerPixel));
		int maxX = std::min((float)mnGridCols, std::ceil((cX + radius) * fGridColPerPixel));
		int minY = std::max(0.f, std::floor((cY - radius) * fGridRowPerPixel));
		int maxY = std::min((float)mnGridRows, std::ceil((cY + radius) * fGridRowPerPixel));

		const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

		for (int y = minY; y < maxY; ++y)
		{
			for (int x = minX; x < maxX; ++x)
			{
				const std::vector<int>& vCell = mvFeaturesInGrid[y][x];

				int nGridSize = mvFeaturesInGrid[y][x].size();
				for (int k = 0; k < nGridSize; ++k)
				{
					const cv::KeyPoint& kp = mvKeysLeft[vCell[k]];
                    if (bCheckLevels)
                    {
                        if (kp.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kp.octave > maxLevel)
                                continue;
                    }

                    const float distx = kp.pt.x - cX;
                    const float disty = kp.pt.y - cY;

                    if (fabs(distx) < radius && fabs(disty) < radius)
                        vCandidates.push_back(vCell[k]);
				}
			}
		}

		return vCandidates;

	}

    void Frame::RecordKeyPointInfo(const float scoreth)
    {
        std::vector<cv::KeyPoint> vSelectedKeys;

        for (auto pt : mvKeysLeft)
        {
            vSelectedKeys.push_back(pt);
        }


        cv::Mat mImgWithInfo;
        Monitor::DrawKeyPointsWithInfo(mRGBLeft, vSelectedKeys, mImgWithInfo, -1.f);
        cv::imshow("mImgWithInfo", mImgWithInfo);
        cv::waitKey(0);
    }

    void Frame::Record()
    {
        cv::Mat matchedImg;
        cv::Mat modifiedImg;
        cv::Mat uncertaintyImg;
        cv::Mat mUFeatureImg;
        cv::Mat mUMatchImg;
        cv::Mat mUAngleImg;
        cv::Mat mUFuseImg;


        std::vector<cv::KeyPoint> vMatchedKeysLeft;
        std::vector<cv::KeyPoint> vMatchedKeysRight;
        std::vector<cv::KeyPoint> vMatchedKeysRightModifiied;
        std::vector<float> vUncertaintyLeft;
        std::vector<float> vUncertaintyRight;
        std::vector<float> vMatchingCost;
        std::vector<float> vAngleRatios;
        std::vector<float> vFuseUncertainty;

		std::vector<std::pair<float, int> > vRawUF;
		std::vector<std::pair<float, int> > vRawUM;
		std::vector<std::pair<float, int> > vRawUA;
		std::vector<std::pair<float, int> > vUFuse;

		int nMatchedPoints = 0;
		for (int i = 0; i < N; ++i)
		{
			if (mvMatches[i] < 0)
				continue;

			vMatchedKeysLeft.push_back(mvKeysLeft[i]);
			vMatchedKeysRight.push_back(mvKeysRight[mvMatches[i]]);

			const float& uF = mvResponse[i];
			const float& uM = mvMatchRatios[i];
			const float& uA = mvAngleRatios[i];
			const float& uFuse = mvFuseUncertainty[i];

			vRawUF.push_back(std::make_pair(uF, i));
			vRawUM.push_back(std::make_pair(uM, i));
			vRawUA.push_back(std::make_pair(uA, i));
			vUFuse.push_back(std::make_pair(uFuse, i));

			nMatchedPoints++;
		}

		std::sort(vRawUA.begin(), vRawUA.end());
		std::sort(vRawUF.begin(), vRawUF.end());
		std::sort(vRawUM.begin(), vRawUM.end());
		std::sort(vUFuse.begin(), vUFuse.end());


		std::vector<cv::KeyPoint> vMatchedKeysLeftF;
		std::vector<cv::KeyPoint> vMatchedKeysRightF;
		std::vector<cv::KeyPoint> vMatchedKeysLeftA;
		std::vector<cv::KeyPoint> vMatchedKeysRightA;
		std::vector<cv::KeyPoint> vMatchedKeysLeftM;
		std::vector<cv::KeyPoint> vMatchedKeysRightM;
		std::vector<cv::KeyPoint> vMatchedKeysLeftFuse;
		std::vector<cv::KeyPoint> vMatchedKeysRightFuse;

		int nMatchedKeys = 0;

		fstream file("uncertainty.txt", ios::in | ios::out);
		if (!file.is_open())
		{
			LOG(ERROR) << "Open file error.";
			return;
		}

		for (int i = 0; i < nMatchedPoints * 0.8; ++i)
		{
			int nReverseI = nMatchedPoints - i - 1;
			vUncertaintyLeft.push_back(vRawUF[nReverseI].first);
			int idxF = vRawUF[nReverseI].second;
			vMatchedKeysLeftF.push_back(mvKeysLeft[idxF]);
			vMatchedKeysRightF.push_back(mvKeysRight[mvMatches[idxF]]);

			vAngleRatios.push_back(vRawUA[nReverseI].first);
			int idxA = vRawUA[nReverseI].second;
			vMatchedKeysLeftA.push_back(mvKeysLeft[idxA]);
			vMatchedKeysRightA.push_back(mvKeysRight[mvMatches[idxA]]);

			vMatchingCost.push_back(vRawUM[i].first);
			int idxM = vRawUM[i].second;
			vMatchedKeysLeftM.push_back(mvKeysLeft[idxM]);
			vMatchedKeysRightM.push_back(mvKeysRight[mvMatches[idxM]]);

			vFuseUncertainty.push_back(vUFuse[nReverseI].first);
			int idxFuse = vRawUA[nReverseI].second;
			vMatchedKeysLeftFuse.push_back(mvKeysLeft[idxFuse]);
			vMatchedKeysRightFuse.push_back(mvKeysRight[mvMatches[idxFuse]]);

			const float response = vUncertaintyLeft[nMatchedKeys];
			const float match = vMatchingCost[nMatchedKeys];
			const float angle = vAngleRatios[nMatchedKeys];
			const float fuse = vFuseUncertainty[nMatchedKeys];

			LOG(INFO) << "Frame: " << mnId <<
					  " Response: " << response << " "
					  << "Match: " << match << " "
					  << "Angle: " << angle << " "
					  << "Fuse: " << fuse;

			file << response << " " << match << " " << angle << " " << fuse << endl;

			nMatchedKeys++;
		}
		file.close();

		/*
        for (int i = 0; i < N; i += 1)
        {
            if (mvMatches[i] < 0) continue;
			else
            {
                const cv::KeyPoint& kpl = mvKeysLeft[i];
                const cv::KeyPoint& kpr = mvKeysRight[mvMatches[i]];

                vMatchedKeysLeft.push_back(kpl);
                vMatchedKeysRight.push_back(kpr);
                vMatchedKeysRightModifiied.push_back(mvKeysRightWithSubPixel[mvMatches[i]]);

				vUncertaintyLeft.push_back(mvResponse[i]);
				vAngleRatios.push_back(mvAngleRatios[i]);
				vMatchingCost.push_back(mvMatchRatios[i]);
				vFuseUncertainty.push_back(mvFuseUncertainty[i]);

				LOG(INFO) << "Frame: " << mnId <<
						  " Keypoint: " << i <<
                          " Response: " << vUncertaintyLeft[nMatchedKeys] << " "
                          << "Match: " << vMatchingCost[nMatchedKeys] << " "
                          << "Angle: " << vAngleRatios[nMatchedKeys] << " "
                          << "Fuse: " << vFuseUncertainty[nMatchedKeys];

                nMatchedKeys++;
            }
        }
        */

        LOG(INFO) << "Frame: " << mnId << " num of matched points: " << nMatchedKeys;

        cv::Mat rgbLeft;
        cv::Mat rgbRight;
        cv::cvtColor(mLeft, rgbLeft, CV_GRAY2RGB);
        cv::cvtColor(mRight, rgbRight, CV_GRAY2RGB);

        // Draw grid
#ifdef DEBUG_DRAW_GRID
         Monitor::DrawGridOnImage(rgbLeft, mnGridRows, mnGridCols, rgbLeft, cv::Scalar(255, 0, 0));
         Monitor::DrawGridOnImage(rgbRight, mnGridRows, mnGridCols, rgbRight, cv::Scalar(255, 0, 0));
#endif

#ifdef DEBUG_DRAW_STEREOMATCHES
        // Monitor::DrawMatchesBetweenStereoFrame(rgbLeft, rgbRight, vMatchedKeysLeft, vMatchedKeysRight, matchedImg, std::vector<float>());
#endif

//		const string tag = "Frame: " + to_string(mnId) + "-";
        cv::Mat keysImg;
        Monitor::DrawKeyPointsOnStereoFrame(rgbLeft, rgbRight, vMatchedKeysLeft, vMatchedKeysRight, keysImg);

//		cv::imwrite(tag + "keys image.png", keysImg);

#ifdef DEBUG_DRAW_MODIFIED
        Monitor::DrawMatchesWithModifiedPosition(rgbLeft, rgbRight, vMatchedKeysLeft,
                                                 vMatchedKeysRight, vMatchedKeysRightModifiied, modifiedImg);
#endif

//        if (mnId > 0)
//            cv::destroyWindow("modified img " + to_string(mnId - 1));

//        cv::imshow("modified img " + to_string(mnId), modifiedImg);
//		cv::imwrite(tag + "modified image.png", modifiedImg);

#ifdef DEBUG_DRAW_UNCERTAINTY


        Monitor::DrawPointsWithUncertaintyByRadius(mRGBLeft, mRGBRight,
                                                   vMatchedKeysLeftF, vMatchedKeysRightF,
                                                   mUFeatureImg, vUncertaintyLeft, std::vector<float>(), true, cv::Scalar(0, 0, 255));

        Monitor::DrawPointsWithUncertaintyByRadius(mRGBLeft, mRGBRight,
                                                   vMatchedKeysLeftM, vMatchedKeysRightM,
                                                   mUMatchImg, vMatchingCost, std::vector<float>(), true, cv::Scalar(255, 0, 0));

        Monitor::DrawPointsWithUncertaintyByRadius(mRGBLeft, mRGBRight,
                                                   vMatchedKeysLeftA, vMatchedKeysRightA,
                                                   mUAngleImg, vAngleRatios, std::vector<float>(), true, cv::Scalar(0, 97, 255));

        Monitor::DrawPointsWithUncertaintyByRadius(mRGBLeft, mRGBRight,
                                                   vMatchedKeysLeftFuse, vMatchedKeysRightFuse, mUFuseImg,
                                                  vFuseUncertainty, std::vector<float>(), true, cv::Scalar(0, 255, 255));

#endif

		string strId = ""; // to_string(mnId);
        if (!matchedImg.empty())
            cv::imshow("matched image " + strId, matchedImg);
        if (!modifiedImg.empty())
            cv::imshow("modified img " + strId, modifiedImg);
        if (!mUFeatureImg.empty())
            cv::imshow("mUFeature " + strId, mUFeatureImg);
        if (!mUMatchImg.empty())
            cv::imshow("mUMatch " + strId, mUMatchImg);
        if (!mUAngleImg.empty())
            cv::imshow("mUAngle " + strId, mUAngleImg);
        if (!mUFuseImg.empty())
            cv::imshow("mUFuse "  + strId, mUFuseImg);

        if (!matchedImg.empty())
            cv::imshow("mKeys " + strId, matchedImg);
		if (!keysImg.empty())
			cv::imshow("keys " + strId, keysImg);


        cv::waitKey(0);

//        cv::imwrite("UFeature_" + to_string(mnId) + ".png", mUFeatureImg);
//        cv::imwrite("UMatch_" + to_string(mnId) + ".png", mUMatchImg);
//        cv::imwrite("UAngle_" + to_string(mnId) + ".png", mUAngleImg);
//        cv::imwrite("UFuse_" + to_string(alphaF) + "_" + to_string(alphaM) + "_" + to_string(alphaA) +
//							"_" + to_string(mnId) + ".png", mUFuseImg);
    }

	void Frame::GenerateDisparityMap()
	{
//        float max = 0;
//        float min = -10000;
		for (int i = 0; i < N; ++i)
		{
			if (mvMatches[i] < 0)
				continue;

			const cv::Point2f& p = mvKeysLeft[i].pt;
			const float d = mvuRight[i] - p.x;

			mDisp.at<float>(p.y, p.x) = abs(d) * 100;

//            if (max < d)
//                max = d;
//            if (min > d)
//                min = d;
		}

        cv::Mat map = mDisp.clone();

        double min;
        double max;
        cv::minMaxIdx(map, &min, &max);
        cv::Mat adjMap;
// expand your range to 0..255. Similar to histEq();
        map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);

// this is great. It converts your grayscale image into a tone-mapped one,
// much more pleasing for the eye
// function is found in contrib module, so include contrib.hpp
// and link accordingly
        cv::Mat falseColorsMap;
        cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);

        cv::imshow("Out", falseColorsMap);

        const std::string tag = "Frame: " + std::to_string(mnId) + "-";
//        cv::imwrite(tag + "depthmap.png", falseColorsMap);
	}

	void Frame::SaveDepthMap()
	{
		std::string path = to_string(mnId) + ".txt";
		ofstream file(path.c_str(), ios::out);
		if (!file.is_open())
		{
			LOG(ERROR) << "Open file error.";
		}

		for (int i = 0; i < N; ++i)
		{
			if (mvMatches[i] < 0)
				continue;

			cv::Mat X3Dw = UnprojectStereo(i);
			const float x = X3Dw.at<float>(0);
			const float y = X3Dw.at<float>(1);
			const float z = X3Dw.at<float>(2);

			file << x << " " << y << " " << z << endl;
		}

		file.close();
	}
}
