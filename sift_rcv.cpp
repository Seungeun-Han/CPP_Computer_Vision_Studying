#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "precomp.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/utils/tls.hpp>

#include "sift.simd.hpp"
#include <opencv2/core/utils/trace.hpp>

using namespace std;
using namespace cv;

// 클래스 정의
class SIFT_hse : public Feature2D {
public:
    explicit SIFT_hse(int nfeatures = 0, int nOctaveLayers = 3,
        double contrastThreshold = 0.04, double edgeThreshold = 10,
        double sigma = 1.6, int descriptorType = CV_32F); //원치 않는 형변환 방지

    //! finds the keypoints and computes descriptors for them using SIFT algorithm.
    //! Optionally it can compute descriptors for the user-provided keypoints
    void detectAndCompute(Mat img, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints = false);

    void buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves);
    void buildDoGPyramid(const vector<Mat>& pyr, vector<Mat>& dogpyr) const;
    void findScaleSpaceExtrema(const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr, vector<KeyPoint>& keypoints) const;

    //descriptor size : 128
    int descriptorSize() {
        return SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; //4*4*8
    }

    int descriptorType() {
        return descriptor_type;
    }

    int defaultNorm() {
        return NORM_L2;
    }

protected:
    int nfeatures;  //feature 개수
    int nOctaveLayers; //옥타브 개수
    double contrastThreshold; //k
    double edgeThreshold; 
    double sigma;
    int descriptor_type;
};

SIFT_hse::SIFT_hse(int _nfeatures, int _nOctaveLayers,
    double _contrastThreshold, double _edgeThreshold, double _sigma, int _descriptorType)
    : nfeatures(_nfeatures), nOctaveLayers(_nOctaveLayers),
    contrastThreshold(_contrastThreshold), edgeThreshold(_edgeThreshold), sigma(_sigma), descriptor_type(_descriptorType)
{
}

// 토대 영상 생성 및 가우시안 블러링
static Mat createInitialImage(const Mat& img, bool doubleImageSize, float sigma){ 
    Mat gray, gray_fpt;
    if (img.channels() == 3 || img.channels() == 4) //컬러면
    {
        cvtColor(img, gray, COLOR_BGR2GRAY); //그레이스케일
        gray.convertTo(gray_fpt, DataType<float>::type); //float 으로
    }
    else
        img.convertTo(gray_fpt, DataType<float>::type); //float 으로

    float sig_diff;

    if (doubleImageSize){ //-1옥타브 만든다면 -> 반복성 크게 향상
        sig_diff = sqrtf(sigma * sigma - 0.5f * 0.5f * 4); //토대 영상은 루트(1.6^2-0.5^2)으로 가우시안블러링
        Mat doubled;

        resize(gray_fpt, doubled, Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, INTER_LINEAR); // 가로 세로 2배

        Mat result;
        GaussianBlur(doubled, result, Size(), sig_diff, sig_diff);
        return result;
    }
    else{
        sig_diff = sqrtf(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
        Mat result;
        GaussianBlur(gray_fpt, result, Size(), sig_diff, sig_diff);
        return result;
    }
}

//가우시안 피라미드
void SIFT_hse::buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves) {
    vector<double> sig(nOctaveLayers + 3);
    pyr.resize(nOctaves * (nOctaveLayers + 3));

    // 시그마 미리 계산
    sig[0] = sigma;
    double k = pow(2., 1. / nOctaveLayers);
    for (int i = 1; i < nOctaveLayers + 3; i++){
        double sig_prev = pow(k, (double)(i - 1)) * sigma;
        double sig_total = sig_prev * k;
        sig[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev); //sigma_{total}^2 = sigma_{i}^2 + sigma_{i-1}^2
    }

    for (int o = 0; o < nOctaves; o++){
        for (int i = 0; i < nOctaveLayers + 3; i++){
            Mat& dst = pyr[o * (nOctaveLayers + 3) + i];
            if (o == 0 && i == 0)
                dst = base; //맨 아래 피라미드는 토대 영상
            else if (i == 0){
                const Mat& src = pyr[(o - 1) * (nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_NEAREST);  //옥타브 올라갈 때 가로세로 1/2 토대 영상 생성
            }
            else{
                const Mat& src = pyr[o * (nOctaveLayers + 3) + i - 1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]); //미리 계산한 시그마로 가우시안 블러링
            }
        }
    }
}

class buildDoGPyramidComputer : public ParallelLoopBody{ //DOG 피라미드: 가우시안 영상 빼기
    public:
        buildDoGPyramidComputer(
            int _nOctaveLayers,
            const vector<Mat>& _gpyr,
            vector<Mat>& _dogpyr)
            : nOctaveLayers(_nOctaveLayers),
            gpyr(_gpyr),
            dogpyr(_dogpyr) { }

        void operator()(const Range& range) const {
            const int begin = range.start;
            const int end = range.end;

            for (int a = begin; a < end; a++) {
                const int o = a / (nOctaveLayers + 2);
                const int i = a % (nOctaveLayers + 2);

                const Mat& src1 = gpyr[o * (nOctaveLayers + 3) + i];
                const Mat& src2 = gpyr[o * (nOctaveLayers + 3) + i + 1];
                Mat& dst = dogpyr[o * (nOctaveLayers + 2) + i];
                subtract(src2, src1, dst, noArray(), DataType<float>::type);
            }
        }

    private:
        int nOctaveLayers;
        const vector<Mat>& gpyr; //가우시안 피라미드
        vector<Mat>& dogpyr; //DOG 피라미드
 };

void SIFT_hse::buildDoGPyramid(const vector<Mat>&gpyr, vector<Mat>&dogpyr) const{
    int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
    dogpyr.resize(nOctaves * (nOctaveLayers + 2));

    //병렬로 계산
    parallel_for_(Range(0, nOctaves * (nOctaveLayers + 2)), buildDoGPyramidComputer(nOctaveLayers, gpyr, dogpyr));
}

//극점 찾기 클래스
class findScaleSpaceExtremaComputer : public ParallelLoopBody {
    public:
        findScaleSpaceExtremaComputer(
            int _o,
            int _i,
            int _threshold,
            int _idx,
            int _step,
            int _cols,
            int _nOctaveLayers,
            double _contrastThreshold,
            double _edgeThreshold,
            double _sigma,
            const vector<Mat>& _gauss_pyr,
            const vector<Mat>& _dog_pyr,
            TLSData<vector<KeyPoint> >& _tls_kpts_struct)
            : o(_o),
            i(_i),
            threshold(_threshold),
            idx(_idx),
            step(_step),
            cols(_cols),
            nOctaveLayers(_nOctaveLayers),
            contrastThreshold(_contrastThreshold),
            edgeThreshold(_edgeThreshold),
            sigma(_sigma),
            gauss_pyr(_gauss_pyr),
            dog_pyr(_dog_pyr),
            tls_kpts_struct(_tls_kpts_struct) { }
        void operator()(const Range& range) const {
            vector<KeyPoint>& kpts = tls_kpts_struct.getRef(); // Get data associated with key

            //메소드 호출
            findScaleSpaceExtrema(o, i, threshold, idx, step, cols, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, gauss_pyr, dog_pyr, kpts, range);
        }
    private:
        int o, i;
        int threshold;
        int idx, step, cols;
        int nOctaveLayers;
        double contrastThreshold;
        double edgeThreshold;
        double sigma;
        const vector<Mat>& gauss_pyr;
        const vector<Mat>& dog_pyr;
        TLSData<vector<KeyPoint> >& tls_kpts_struct;
};

// Bad features are discarded based on contrast and ratio of principal curvatures.
// 극점 검출 메소드
void SIFT_hse::findScaleSpaceExtrema(const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
    vector<KeyPoint>& keypoints) const {
    const int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
    const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * 1);

    keypoints.clear();
    TLSDataAccumulator<vector<KeyPoint> > tls_kpts_struct;

    for (int o = 0; o < nOctaves; o++)
        for (int i = 1; i <= nOctaveLayers; i++)
        {
            const int idx = o * (nOctaveLayers + 2) + i;
            const Mat& img = dog_pyr[idx];
            const int step = (int)img.step1();
            const int rows = img.rows, cols = img.cols;

            parallel_for_(Range(SIFT_IMG_BORDER, rows - SIFT_IMG_BORDER), //5x5
                findScaleSpaceExtremaComputer(
                    o, i, threshold, idx, step, cols,
                    nOctaveLayers,
                    contrastThreshold,
                    edgeThreshold,
                    sigma,
                    gauss_pyr, dog_pyr, tls_kpts_struct));
        }

    vector<vector<KeyPoint>*> kpt_vecs;
    tls_kpts_struct.gather(kpt_vecs); 
    for (size_t i = 0; i < kpt_vecs.size(); ++i) {
        keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end()); 
    }
}

static inline void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale){
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}

class calcDescriptorsComputer : public ParallelLoopBody{
public:
    calcDescriptorsComputer(const vector<Mat>& _gpyr,
        const vector<KeyPoint>& _keypoints,
        Mat& _descriptors,
        int _nOctaveLayers,
        int _firstOctave)
        : gpyr(_gpyr),
        keypoints(_keypoints),
        descriptors(_descriptors),
        nOctaveLayers(_nOctaveLayers),
        firstOctave(_firstOctave) { }

    void operator()(const Range& range) const {
        const int begin = range.start;
        const int end = range.end;

        static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS; //4, 8

        for (int i = begin; i < end; i++){
            KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;
            unpackOctave(kpt, octave, layer, scale);
            CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
            float size = kpt.size * scale;
            Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);
            const Mat& img = gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer];

            float angle = 360.f - kpt.angle;
            if (std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;
            cv::calcSIFTDescriptor(img, ptf, angle, size * 0.5f, d, n, descriptors, i);
        }
    }
private:
    const vector<Mat>& gpyr;
    const vector<KeyPoint>& keypoints;
    Mat& descriptors;
    int nOctaveLayers;
    int firstOctave;
};

static void calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
    Mat& descriptors, int nOctaveLayers, int firstOctave) {
    parallel_for_(Range(0, static_cast<int>(keypoints.size())), calcDescriptorsComputer(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave));
}

//SIFT 알고리즘 전체 실행
void SIFT_hse::detectAndCompute(Mat _image, vector<KeyPoint>& keypoints, OutputArray _descriptors, bool useProvidedKeypoints) {
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image = _image;

    if (image.empty() || image.depth() != CV_8U)
        CV_Error(Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

    if (useProvidedKeypoints) {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for (size_t i = 0; i < keypoints.size(); i++) {
            int octave, layer;
            float scale;
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = min(firstOctave, octave);
            maxOctave = max(maxOctave, octave);
            actualNLayers = max(actualNLayers, layer - 2);
        }

        firstOctave = min(firstOctave, 0);
        CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
    vector<Mat> gpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

    buildGaussianPyramid(base, gpyr, nOctaves);

    if (!useProvidedKeypoints) {
        vector<Mat> dogpyr;
        buildDoGPyramid(gpyr, dogpyr);
        findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
        KeyPointsFilter::removeDuplicatedSorted(keypoints);

        if (nfeatures > 0)
            KeyPointsFilter::retainBest(keypoints, nfeatures);

        if (firstOctave < 0)
            for (size_t i = 0; i < keypoints.size(); i++)
            {
                KeyPoint& kpt = keypoints[i];
                float scale = 1.f / (float)(1 << -firstOctave);
                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                kpt.pt *= scale;
                kpt.size *= scale;
            }
    }

    if (_descriptors.needed())
    {
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, descriptor_type);

        Mat descriptors = _descriptors.getMat();
        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
    }
    
}


int main(int argc, char** argv) {
	Mat image = imread("box.png", IMREAD_GRAYSCALE);  //figures.jpg / blox.jpg / box.png / corner.jpg
	int Octave = 3;
    double k = 0.04;
    double edgeThreshold = 10;
	double sigma = 1.6;

    //라이브러리
	Ptr<SIFT>  siftPtr = SIFT::create(0, Octave, k, edgeThreshold, sigma);
	vector<KeyPoint> keypoints;
	siftPtr->detect(image, keypoints);
	Mat cv_library;
	drawKeypoints(image, keypoints, cv_library);

    //SIFT_hse
    SIFT_hse  SIFT_Impl_Ptr = SIFT_hse(0, Octave, k, edgeThreshold, sigma);
    vector<KeyPoint> SIFT_Impl_keypoints;
    Mat output;
    SIFT_Impl_Ptr.detectAndCompute(image, SIFT_Impl_keypoints, output);
    Mat SIFT_hse_result;
    drawKeypoints(image, SIFT_Impl_keypoints, SIFT_hse_result);

	imshow("image", image);
	imshow("cv_library", cv_library);
    imshow("SIFT_hse", SIFT_hse_result);
    imshow("output", output);
	waitKey();
	return 0;
}