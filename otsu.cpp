#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void calc_histo(Mat image, Mat& hist, int bins, int range_max = 256) {
	hist = Mat(bins, 1, CV_32F, Scalar(0));
	float gap = range_max / (float)bins;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int idx = int(image.at<uchar>(i, j) / gap);
			hist.at<float>(idx)++;
		}
	}
}

void draw_histo(Mat hist, Mat& hist_img, Size size = Size(256, 256)) {
	hist_img = Mat(size, CV_8U, Scalar(255));
	float bin = (float)hist_img.cols / hist.rows;
	normalize(hist, hist, 0, hist_img.rows, NORM_MINMAX);

	for (int i = 0; i < hist.rows; i++) {
		float start_x = i * bin;
		float end_x = (i + 1) * bin;
		Point2f pt1(start_x, 0);
		Point2f pt2(end_x, hist.at<float>(i));

		if (pt2.y > 0) {
			rectangle(hist_img, pt1, pt2, Scalar(0), -1);
		}
	}
	flip(hist_img, hist_img, 0);
}

int Calc_Thres_by_Otsu(Mat image) {
	Mat hist;
	double m = 0;

	calc_histo(image, hist, 256, 256);

	for (int i = 0; i < hist.rows; i++) { //normalize
		hist.at<float>(i) = hist.at<float>(i) / (image.rows * image.cols);
	}

	for (int i = 0; i < hist.rows; i++) {
		m += (double)i * (double)hist.at<float>(i);
	}

	double w0[256] = { 0. };
	double m0[256] = { 0. };
	double m1[256] = { 0. };
	double v_bet[256] = { 0. };
	w0[0] = (double)hist.at<float>(0);
	m0[0] = 0.0;

	for (int t = 1; t < 256; t++) {
		w0[t] = w0[t - 1] + (double)hist.at<float>(t);
		if (w0[t] == 0) {
			continue;
		}
		m0[t] = (w0[t - 1] * m0[t - 1] + (double)t * (double)hist.at<float>(t)) / w0[t];
		m1[t] = (m - w0[t] * m0[t]) / (1 - w0[t]);
		v_bet[t] = w0[t] * (1 - w0[t]) * ((m0[t] - m1[t]) * (m0[t] - m1[t]));
	}

	double max = v_bet[1];
	int thres = 1;
	for (int i = 2; i < 256; i++) {
		if (v_bet[i] > max) {
			max = v_bet[i];
			thres = i;
		}
	}

	return thres;
}

int main() {
	Mat image = imread("bright.jpeg", IMREAD_GRAYSCALE);
	Mat otsu(image.size(), image.type(), Scalar(0));
	Mat cv_otsu;
	int thres = 0;

	thres = Calc_Thres_by_Otsu(image);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) >= thres) {
				otsu.at<uchar>(i, j) = 255;
			}
		}
	}

	cout << "thresold: " << thres << endl;
	
	int cv_thres = threshold(image, cv_otsu, 0, 255, THRESH_BINARY | THRESH_OTSU);
	cout << "cv_thresold: " << cv_thres << endl;

	imshow("original image", image);
	imshow("result", otsu);
	imshow("cv_result", cv_otsu);
	imwrite("otsu.jpg", cv_otsu); 
	waitKey();
	return 0;
}