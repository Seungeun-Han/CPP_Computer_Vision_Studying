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

Mat scramble(Mat signal) {
	Mat dst = signal.clone();
	for (int i = 0, j = 0; i < dst.cols - 1; i++) {
		if (i > j) {
			swap(dst.at<Vec2f>(i), dst.at<Vec2f>(j));
		}

		int m = dst.cols >> 1;
		while ((j >= m)&&(m >= 2)) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}
	return dst;
}

Mat zeropadding(Mat img) {
	int m = 1 << (int)ceil(log2(img.rows));
	int n = 1 << (int)ceil(log2(img.cols));
	Mat dst(m, n, img.type(), Scalar(0));

	Rect rect(Point(0, 0), img.size());
	img.copyTo(dst(rect));
	dst.convertTo(dst, CV_32F);
	return dst;
}

void butterfly(Mat& dst, int dir) {
	int length = dst.cols;
	int pair = 1;
	for (int k = 0; k < ceil(log2(length)); k++) {
		int half_pair = pair;
		pair <<= 1;
		float theta = dir * ((-2.0) * CV_PI / pair);
		float wpr = (-2.0) * sin(0.5 * theta) * sin(0.5 * theta);
		float wpi = sin(theta);
		float wre = 1.0;
		float wim = 0.0;

		for (int m = 0; m < half_pair; m++) {
			for (int even = m; even < length; even += pair) {
				int odd = even + half_pair;
				Vec2f G_even = dst.at<Vec2f>(even);
				Vec2f G_odd = dst.at<Vec2f>(odd);

				Vec2f G_odd_W(0, 0);
				G_odd_W[0] = G_odd[0] * wre - G_odd[1] * wim;
				G_odd_W[1] = G_odd[1] * wre - G_odd[0] * wim;

				dst.at<Vec2f>(even) = G_even + G_odd_W;
				dst.at<Vec2f>(odd) = G_even - G_odd_W;
			}
			float tmp = wre;
			wre += tmp * wpr - wim * wpi;
			wim += wim * wpr + tmp * wpi;
		}
	}
	if (dir == -1) {
		dst /= dst.cols;
	}
}

void FFT_2D(Mat complex_img, Mat& dst, int dir) {
	dst = Mat(complex_img.size(), complex_img.type());
	for (int i = 0; i < complex_img.rows; i++) {
		Mat scr_sn = scramble(complex_img.row(i));
		butterfly(scr_sn, dir);
		scr_sn.copyTo(dst.row(i));
	}

	transpose(dst, dst);
	for (int i = 0; i < dst.rows; i++) {
		Mat scr_sn = scramble(dst.row(i));
		butterfly(scr_sn, dir);
		scr_sn.copyTo(dst.row(i));
	}
	transpose(dst, dst);
}

int main() {
	Mat complex_img, idft_img, img_tmp[2];
	Mat dft_coef1, dft_img1, shuffle_img1;
	Mat dft_coef2, dft_img2, shuffle_img2;
	Mat image = imread("Lenna.png", 0);
	Mat pad_img = zeropadding(image);
	Mat tmp[] = { pad_img, Mat::zeros(pad_img.size(), pad_img.type()) };
	merge(tmp, 2, complex_img);

	dft(complex_img, dft_coef2, 1);
	log_mag(dft_coef2, dft_img2);
	shuffling(dft_img2, shuffle_img2);

	FFT_2D(complex_img, dft_coef1, 1);
	log_mag(dft_coef1, dft_img1);
	shuffling(dft_img2, shuffle_img1);

	imshow("image", image);
	imshow("shuffling", shuffle_img1);
	imshow("shuffling_opencv", shuffle_img2);

	imwrite("fft.jpg", shuffle_img1);
	imwrite("fft_opencv.jpg", shuffle_img2);
	waitKey();
	return 0;
}