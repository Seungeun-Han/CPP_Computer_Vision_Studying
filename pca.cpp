#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

int main() {
	//Mat imageS[10];
	/*for (int i = 0; i < 10; i++) {
		imageSet[i] = imread("pcadata/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		imageSet[i].convertTo(imageSet[i], CV_32FC1);
	}*/
	vector<Mat> imageSet;
	
	for (int i = 0; i < 10; i++) {
		//imageSet.push_back(imread("pcadata/" + to_string(i) + ".jpg", 0));
		Mat image = imread("pcadata/" + to_string(i) + ".jpg", 0);
		image.convertTo(image, CV_32FC1);
		image = image.reshape(1, image.rows * image.cols);
		//vector<float> vecImg;
		//vecImg.assign(image.datastart, image.dataend);
		
		imageSet.push_back(image.t());
		//imageSet.push_back(image);
		//imageS[i] = image;
	}
	//cout << imageSet.size() << endl;
	cout << imageSet[0].size() << endl;

	Mat avgImage = imageSet[0].clone();
	for (int i = 1; i < 10;i++) {
		avgImage += imageSet[i];
	}
	avgImage = avgImage / 10.0;
	cout << avgImage.size() << endl;
	//cout << avgImage << endl;

	/*avgImage = avgImage.reshape(1, 96);
	avgImage.convertTo(avgImage, CV_8UC1);
	imwrite("avg.jpg", avgImage);*/

	// 원점 중심으로 (원본영상 - 평균영상)
	for (int i = 0; i < 10; i++) {
		imageSet[i] = imageSet[i] - avgImage;
	}
	//cout << imageSet[5] << endl;

	// 공분산 행렬
	Mat covariance;
	calcCovarMatrix(imageSet, covariance, avgImage, COVAR_ROWS, CV_32FC1);
	//calcCovarMatrix(imageSet, covariance, avgImage, COVAR_NORMAL, CV_32FC1);
	//covariance.convertTo(covariance, CV_32FC1);
	cout << covariance.size() << endl;
	//cout << covariance<< endl;
	cout << endl;

	// 고유값, 고유벡터
	Mat eigenvalues, eigenvectors;
	eigen(covariance, eigenvalues, eigenvectors);
	cout << eigenvalues.size() << endl;
	//cout << eigenvalues << endl;
	cout << endl;
	cout << "vector"<<eigenvectors.size() << endl;
	//cout << eigenvectors << endl;

	//float imageS[10][96*96];
	Mat imageS(10, 96 * 96, CV_32FC1);
	for (int i = 0; i < imageS.rows; i++) {
		for (int j = 0; j < imageS.cols; j++) {
			imageS.at<float>(i, j) = imageSet[i].at<float>(j);
		}
	}
	cout << imageS.size() << endl;
	Mat dst;// = Mat::zeros(imageSet[0].size(), imageSet[0].type());

	gemm(imageS.t(), eigenvectors, 1, Mat(), 0, dst);
	//cout << "dst" << dst << endl;
	//cout << "dst" << dst.size() << endl;
	Mat rank1 = dst.t().row(0);
	
	rank1 = rank1.reshape(1, 96);
	cout << rank1.size() << endl;
	rank1.convertTo(rank1, CV_8UC1);

	Mat rank2 = dst.t().row(1);

	rank2 = rank2.reshape(1, 96);
	cout << rank2.size() << endl;
	rank2.convertTo(rank2, CV_8UC1);
	
	Mat rank3 = dst.t().row(2);

	rank3 = rank3.reshape(1, 96);
	cout << rank3.size() << endl;
	rank3.convertTo(rank3, CV_8UC1);
	/*
	int d = 6;
	Mat tmp = eigenvectors.row(0);
	//tmp.convertTo(eigenvectors.at<Mat>(0), CV_8UC1);

	cout << tmp << endl;*/
	
	imwrite("rank1.jpg", rank1);
	imwrite("rank2.jpg", rank2);
	imwrite("rank3.jpg", rank3);
	waitKey();
	return 0;
}