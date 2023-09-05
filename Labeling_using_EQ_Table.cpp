// Copyright (c) 한승은. All rights reserved.

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <ctime>
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
	Mat image = imread("letter.jpg", IMREAD_GRAYSCALE); //figure  //figures  //letter
	Mat otsu(image.size(), image.type(), Scalar(0));
	Mat result(image.size(), CV_32F, Scalar(0));
	Mat cv_otsu;
	int thres = 0;

	thres = Calc_Thres_by_Otsu(image);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) < thres) {
				otsu.at<uchar>(i, j) = 255;
			}
		}
	}
	cout << "thresold: " << thres << endl;

	int label_count = 0;
	int* eq_table = new int[image.rows * image.cols + 1];
	/*2차원 배열 동적할당
	int** eq_table = new int*[image.rows * image.cols + 1]; //0번 인덱스는 사용 안해서 +1
	for (int i = 0; i < image.rows * image.cols + 1; i++) {
		eq_table[i] = new int[2];
	}*/
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (otsu.at<uchar>(i, j) > 0) {
				if (i == 0 && j == 0) {  //(0,0)
					label_count++;
					result.at<float>(i, j) = label_count;
					eq_table[label_count] = label_count;
				}
				else if (i == 0 && j != 0) {  //0행
					if (otsu.at<uchar>(i, j - 1) > 0) {
						result.at<float>(i, j) = result.at<float>(i, j - 1);
					}
					else {
						label_count++;
						result.at<float>(i, j) = label_count;
						eq_table[label_count] = label_count;
					}
				}
				else if (i != 0 && j == 0) {  //0열
					if (otsu.at<uchar>(i - 1, j) > 0) {
						result.at<float>(i, j) = result.at<float>(i - 1, j);
					}
					else {
						label_count++;
						result.at<float>(i, j) = label_count;
						eq_table[label_count] = label_count;
					}
				}

				//서, 북서, 북, 북동 다 없을 때
				if ((i != 0 && j != 0) && otsu.at<uchar>(i - 1, j) == 0
					&& otsu.at<uchar>(i, j - 1) == 0 && otsu.at<uchar>(i - 1, j - 1) == 0
					&& otsu.at<uchar>(i - 1, j + 1) == 0) {
					label_count++;
					result.at<float>(i, j) = label_count;
					eq_table[label_count] = label_count;
				}
				else {  //그 외
					int values[4] = { 0, };
					int n = 0;
					if (otsu.at<uchar>(i - 1, j) > 0) {
						values[n] = result.at<float>(i - 1, j);
						n++;
					}
					if (otsu.at<uchar>(i, j - 1) > 0) {
						values[n] = result.at<float>(i, j - 1);
						n++;
					}
					if (otsu.at<uchar>(i - 1, j - 1) > 0) {
						values[n] = result.at<float>(i - 1, j - 1);
						n++;
					}
					if (otsu.at<uchar>(i - 1, j + 1) > 0) {
						values[n] = result.at<float>(i - 1, j + 1);
						n++;
					}

					int min = values[0];
					for (int k = 0; k < n && values[k] != 0; k++) {
						if (min >= values[k]) {
							min = values[k];
						}
					}

					result.at<float>(i, j) = min;
					if (otsu.at<uchar>(i - 1, j) > 0) {
						eq_table[(int)result.at<float>(i - 1, j)] = min;
						result.at<float>(i - 1, j) = min;
					}
					if (otsu.at<uchar>(i, j - 1) > 0) {
						eq_table[(int)result.at<float>(i, j - 1)] = min;
						result.at<float>(i, j - 1) = min;
					}
					if (otsu.at<uchar>(i - 1, j - 1) > 0) {
						eq_table[(int)result.at<float>(i - 1, j - 1)] = min;
						result.at<float>(i - 1, j - 1) = min;
					}
					if (otsu.at<uchar>(i - 1, j + 1) > 0) {
						eq_table[(int)result.at<float>(i - 1, j + 1)] = min;
						result.at<float>(i - 1, j + 1) = min;
					}
				}
			}
		}
	}
	
	/*eq table 정리*/
	for (int i = 1; i <= label_count; i++) {
		int n = i; 
		while (eq_table[n] != n) {
			n = eq_table[n];
		}
		eq_table[i] = n;
	}

	/*int max = 0;
	for (int i = 1; i <= label_count; i++) {
		if (max < eq_table[i]) {
			max = eq_table[i];
		}
	}
	cout << "max: " << max << endl;*/
	/*int count = eq_table[1];
	int before = 0;
	for (int i = 1; i <= label_count; i++) {
		if (count != eq_table[i] && before != eq_table[i]) {
			before = eq_table[i];
			eq_table[i] = count;
		}
		else if(count == eq_table[i]){
			count++;
		}
	}
	cout << count << endl;*/


	/*변경된 eq_table을 이용하여 다시 labeling*/
	for (int k = 1; k <= label_count; k++) {
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				if (result.at<float>(i, j) != 0 && k != eq_table[k] && result.at<float>(i, j) == k) {
					result.at<float>(i, j) = eq_table[k];
				}
			}
		}
	}

	srand((unsigned int)time(NULL));
	Mat tmp(image.size(), CV_8UC3, Scalar(0, 0, 0));
	for (int k = 1; k <= label_count; k++) {
		Vec3b random_Color = Vec3b((int)rand() % 256, (int)rand() % 256, (int)rand() % 256);
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				if (result.at<float>(i, j) == k) {
					tmp.at<Vec3b>(i, j) = random_Color;
				}
			}
		}
	}
	
	int* area= new int[image.rows * image.cols + 1];
	int* xsum = new int[image.rows * image.cols + 1];
	int* ysum = new int[image.rows * image.cols + 1];

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (result.at<float>(i, j) != 0) {
				area[(int)result.at<float>(i, j)]++;
				xsum[(int)result.at<float>(i, j)] += j;
				ysum[(int)result.at<float>(i, j)] += i;
			}
		}
	}

	/*cout << "eq_table: " << endl;
	for (int i = 1; i <= label_count; i++) {
		cout << i << ": " << eq_table[i] << endl;
	}*/

	label_count = 0;

	// imwrite("image.png", image);
	imwrite("otsu.png", otsu);
	imwrite("tmp.png", tmp);

	imshow("image", image);
	imshow("otsu", otsu);
	imshow("tmp", tmp);
	waitKey();

	delete[] ysum;
	delete[] xsum;
	delete[] area;
	delete[] eq_table;
	return 0;
}
