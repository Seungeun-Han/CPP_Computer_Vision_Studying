// Copyright (c) 한승은. All rights reserved.

#include <opencv2/opencv.hpp>
#include <math.h>
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

/*void calc_histo(Mat image, Mat& hist, int range) {
	hist = Mat(range, 1, CV_32F, Scalar(0));

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			hist.at<float>(int(image.at<uchar>(i, j)))++;
		}
	}
}

void draw_Histo(Mat hist, Mat& hist_image, int range) {
	int max = 0;
	for (int i = 0; i < hist.rows; i++) {
		if (hist.at<float>(i) > max)
			max = hist.at<float>(i);
	}

	for (int i = 0; i < hist.rows; i++) {
		hist.at<float>(i) = (hist.at<float>(i) / max) * (range - 1);
	}

	hist_image = Mat(range, range, CV_8U, Scalar(range - 1));
	for (int i = 0; i < hist.rows; i++) {
		Point2f pt1(i, 0);
		Point2f pt2(i + 1, hist.at<float>(i));
		rectangle(hist_image, pt1, pt2, Scalar(0), -1);
	}
	flip(hist_image, hist_image, 0);
}*/

void otsu_2thres_2th(Mat image, Mat& result) {
	Mat hist;
	double m = 0;
	int thres1 = 0;
	int thres2 = 0;

	calc_histo(image, hist, 256);

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
	for (int i = 2; i < 256; i++) {
		if (v_bet[i] > max) {
			max = v_bet[i];
			thres1 = i;
		}
	}

	double w0_2th[256] = { 0. };
	double m0_2th[256] = { 0. };
	double m1_2th[256] = { 0. };
	double v_bet_2th[256] = { 0. };
	double m_2th = 0.0;
	if (thres1 > 256 / 2) {
		w0_2th[0] = (double)hist.at<float>(0);
		m0_2th[0] = 0.0;

		for (int i = 0; i < thres1; i++) {
			m_2th += (double)i * (double)hist.at<float>(i);
		}

		for (int t = 1; t < thres1; t++) {
			w0_2th[t] = w0_2th[t - 1] + (double)hist.at<float>(t);
			if (w0_2th[t] == 0) {
				continue;
			}
			m0_2th[t] = (w0_2th[t - 1] * m0_2th[t - 1] + (double)t * (double)hist.at<float>(t)) / w0_2th[t];
			m1_2th[t] = (m_2th - w0_2th[t] * m0_2th[t]) / (w0[thres1] - w0_2th[t]);
			v_bet_2th[t] = w0_2th[t] * (w0[thres1] - w0_2th[t]) * ((m0_2th[t] - m1_2th[t]) * (m0_2th[t] - m1_2th[t]));
		}
		max = v_bet_2th[1];
		for (int i = 2; i < thres1; i++) {
			if (v_bet_2th[i] > max) {
				max = v_bet_2th[i];
				thres2 = i;
			}
		}
	}
	else {
		w0_2th[thres1] = (double)hist.at<float>(thres1);
		m0_2th[thres1] = (double)thres1 * (double)hist.at<float>(thres1) / w0_2th[thres1];

		for (int i = thres1; i < 256; i++) {
			m_2th += (double)i * (double)hist.at<float>(i);
		}

		for (int t = thres1 + 1; t < 256; t++) {
			w0_2th[t] = w0_2th[t - 1] + (double)hist.at<float>(t);
			if (w0_2th[t] == 0) {
				continue;
			}
			m0_2th[t] = (w0_2th[t - 1] * m0_2th[t - 1] + (double)t * (double)hist.at<float>(t)) / w0_2th[t];
			m1_2th[t] = (m_2th - w0_2th[t] * m0_2th[t]) / ((1 - w0[thres1]) - w0_2th[t]);
			v_bet_2th[t] = w0_2th[t] * ((1 - w0[thres1]) - w0_2th[t]) * ((m0_2th[t] - m1_2th[t]) * (m0_2th[t] - m1_2th[t]));
		}
		max = v_bet_2th[thres1];
		for (int i = thres1 + 1; i < 256; i++) {
			if (v_bet_2th[i] > max) {
				max = v_bet_2th[i];
				thres2 = i;
			}
		}		
	}

	if (thres1 > thres2) {
		int tmp = 0;
		tmp = thres2;
		thres2 = thres1;
		thres1 = tmp;
	}

	cout << thres1 << " " << thres2 << endl;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) < thres1) {
				result.at<uchar>(i, j) = 0;
			}
			else if (image.at<uchar>(i, j) >= thres1 && image.at<uchar>(i, j) < thres2) {
				result.at<uchar>(i, j) = 127;
			}
		}
	}
}
                           
int main() {
	Mat image = imread("bright.jpeg", IMREAD_GRAYSCALE); 
	Mat result(image.size(), image.type(), Scalar(255));
	
	otsu_2thres_2th(image, result);

	imshow("image", image);
	imshow("result", result);
	waitKey();

	return 0;
}
