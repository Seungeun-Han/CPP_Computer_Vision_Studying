# Copyright (c) 한승은. All rights reserved.

#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;

void log_mag(Mat complex, Mat& dst) { //푸리에 변환을 수행하면 복소수의 행렬이 결과로 생성
	Mat planes[2]; //복소수의 실수부와 허수부를 벡터로 간주
	split(complex, planes); //2채널 행렬 분리
	magnitude(planes[0], planes[1], dst); //벡터의 크기
	log(dst + 1, dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX); //정규화(저주파 영역과 고주파 영역의 계수값을 정규화)
	dst.convertTo(dst, CV_8U);
}

void shuffling(Mat mag_img, Mat& dst) {
	int cx = mag_img.cols / 2;
	int cy = mag_img.rows / 2;
	Rect q1(cx, 0, cx, cy); //1사분면
	Rect q2(0, 0, cx, cy); //2
	Rect q3(0, cy, cx, cy); //3
	Rect q4(cx, cy, cx, cy); //4

	dst = Mat(mag_img.size(), mag_img.type());
	mag_img(q1).copyTo(dst(q3)); //사분면 맞바꿈
	mag_img(q3).copyTo(dst(q1));
	mag_img(q2).copyTo(dst(q4));
	mag_img(q4).copyTo(dst(q2));
}

Mat DFT_1D(Mat row, int dir) { //1차원 신호의 이산 푸리에 변환
	int n = row.cols;
	Mat dst(row.size(), CV_32FC2);

	for (int i = 0; i < n; i++) {
		Vec2f complex(0, 0);
		for (int j = 0; j < n; j++) {
			float theta = dir * (-2) * CV_PI * i * j / n; //기저함수 각도 계산
			Vec2f value = row.at<Vec2f>(j);
			complex[0] += value[0] * cos(theta) - value[1] * sin(theta);
			complex[1] += value[1] * cos(theta) + value[0] * sin(theta);
		}
		dst.at<Vec2f>(i) = complex;  //한 원소의 DFT 계산 결과 저장
	}

	if (dir == -1) { //-1이면 역변환
		dst /= n;
	}

	return dst;
}

void DFT_2D(Mat complex, Mat& dst, int dir) { //2차원 신호 푸리에 변환
	complex.convertTo(complex, CV_32F);
	Mat tmp(complex.size(), CV_32FC2, Vec2f(0, 0));
	tmp.copyTo(dst);

	for (int i = 0; i < complex.rows; i++){  //가로 방향 푸리에 변환
		Mat one_row = complex.row(i);
		Mat dft_row = DFT_1D(one_row, dir);  //1개 행 변환
		dft_row.copyTo(tmp.row(i)); //저장
	}

	transpose(tmp, tmp); //전치
	for (int i = 0; i < tmp.rows; i++) {  //세로 방향 푸리에 변환
		Mat one_row = tmp.row(i);
		Mat dft_row = DFT_1D(tmp.row(i), dir); 
		dft_row.copyTo(dst.row(i));
	}
	transpose(dst, dst);
}

int main() {
	Mat image = imread("Lenna.png", 0);
	Mat complex, dft_coef, dft_img, idft_coef, shuffle, idft_img[2];  
	Mat tmp[] = { image, Mat::zeros(image.size(), CV_8U) };
	merge(tmp, 2, complex); //복소수 행렬 구성

	DFT_2D(complex, dft_coef, 1); //2차원 DFT
	log_mag(dft_coef, dft_img);
	shuffling(dft_img, shuffle);

	DFT_2D(dft_coef, idft_coef, -1); // 푸리에 역변환으로 원본 영상 복원
	split(idft_coef, idft_img);
	idft_img[0].convertTo(idft_img[0], CV_8U);

	imshow("image", image);
	imshow("dft_img", dft_img);
	imshow("shuffling", shuffle);
	imshow("idft_img", idft_img[0]);

	imwrite("dft.jpg", shuffle);
	imwrite("idft_img.jpg", idft_img[0]);
	waitKey();
	return 0;
}
