#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;

void log_mag(Mat complex, Mat& dst) { //Ǫ���� ��ȯ�� �����ϸ� ���Ҽ��� ����� ����� ����
	Mat planes[2]; //���Ҽ��� �Ǽ��ο� ����θ� ���ͷ� ����
	split(complex, planes); //2ä�� ��� �и�
	magnitude(planes[0], planes[1], dst); //������ ũ��
	log(dst + 1, dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX); //����ȭ(������ ������ ������ ������ ������� ����ȭ)
	dst.convertTo(dst, CV_8U);
}

void shuffling(Mat mag_img, Mat& dst) {
	int cx = mag_img.cols / 2;
	int cy = mag_img.rows / 2;
	Rect q1(cx, 0, cx, cy); //1��и�
	Rect q2(0, 0, cx, cy); //2
	Rect q3(0, cy, cx, cy); //3
	Rect q4(cx, cy, cx, cy); //4

	dst = Mat(mag_img.size(), mag_img.type());
	mag_img(q1).copyTo(dst(q3)); //��и� �¹ٲ�
	mag_img(q3).copyTo(dst(q1));
	mag_img(q2).copyTo(dst(q4));
	mag_img(q4).copyTo(dst(q2));
}

Mat DFT_1D(Mat row, int dir) { //1���� ��ȣ�� �̻� Ǫ���� ��ȯ
	int n = row.cols;
	Mat dst(row.size(), CV_32FC2);

	for (int i = 0; i < n; i++) {
		Vec2f complex(0, 0);
		for (int j = 0; j < n; j++) {
			float theta = dir * (-2) * CV_PI * i * j / n; //�����Լ� ���� ���
			Vec2f value = row.at<Vec2f>(j);
			complex[0] += value[0] * cos(theta) - value[1] * sin(theta);
			complex[1] += value[1] * cos(theta) + value[0] * sin(theta);
		}
		dst.at<Vec2f>(i) = complex;  //�� ������ DFT ��� ��� ����
	}

	if (dir == -1) { //-1�̸� ����ȯ
		dst /= n;
	}

	return dst;
}

void DFT_2D(Mat complex, Mat& dst, int dir) { //2���� ��ȣ Ǫ���� ��ȯ
	complex.convertTo(complex, CV_32F);
	Mat tmp(complex.size(), CV_32FC2, Vec2f(0, 0));
	tmp.copyTo(dst);

	for (int i = 0; i < complex.rows; i++){  //���� ���� Ǫ���� ��ȯ
		Mat one_row = complex.row(i);
		Mat dft_row = DFT_1D(one_row, dir);  //1�� �� ��ȯ
		dft_row.copyTo(tmp.row(i)); //����
	}

	transpose(tmp, tmp); //��ġ
	for (int i = 0; i < tmp.rows; i++) {  //���� ���� Ǫ���� ��ȯ
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
	merge(tmp, 2, complex); //���Ҽ� ��� ����

	DFT_2D(complex, dft_coef, 1); //2���� DFT
	log_mag(dft_coef, dft_img);
	shuffling(dft_img, shuffle);

	DFT_2D(dft_coef, idft_coef, -1); // Ǫ���� ����ȯ���� ���� ���� ����
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