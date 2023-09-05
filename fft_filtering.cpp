#include <opencv2/opencv.hpp>
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

Mat zeropadding(Mat img) {  // �Է� ���� �������� �����ؼ� ��ȯ�ϴ� �Լ�
	int m = 1 << (int)ceil(log2(img.rows));  // 2�� �ڽ� ���
	int n = 1 << (int)ceil(log2(img.cols));
	Mat dst(m, n, img.type(), Scalar(0));

	Rect rect(Point(0, 0), img.size());  // ���� ���� ũ�� ���� ����
	img.copyTo(dst(rect));  // ���� ������ ���� ������ ����
	dst.convertTo(dst, CV_32F);
	return dst;
}

Mat get_lowpassFilter(Size size, int radius) {  // ������ ��� ���� ���� �Լ�
	Point center = size / 2;  // �߽��� ����
	Mat filter(size, CV_32FC2, Vec2f(0, 0));  // 2ä�� ��� ����
	circle(filter, center, radius, Vec2f(1, 1), -1);
	/*Mat filter2(size, CV_8UC1, Scalar(0));
	circle(filter2, center, radius, Scalar(255), -1);
	imwrite("lowpass_filter.jpg", filter2);*/
	return filter;
}

Mat get_highpassFilter(Size size, int radius) {  // ������ ��� ���� ���� �Լ�
	Point center = size / 2;
	Mat filter(size, CV_32FC2, Vec2f(1, 1));  
	circle(filter, center, radius, Vec2f(0, 0), -1);  
	/*Mat filter2(size, CV_8UC1, Scalar(255));
	circle(filter2, center, radius, Scalar(0), -1);
	imwrite("highpass_filter.jpg", filter2); */
	return filter;
}

void FFT(Mat image, Mat& dft_coef, Mat& dft_img) {
	Mat complex_img;
	Mat pad_img = zeropadding(image);
	Mat tmp[] = { pad_img, Mat::zeros(pad_img.size(), pad_img.type()) };
	merge(tmp, 2, complex_img);  // ���� ��� ����
	dft(complex_img, dft_coef, 0);
	shuffling(dft_coef, dft_coef);
	log_mag(dft_coef, dft_img);  // ���ļ� ����Ʈ�� ����
}

Mat IFFT(Mat dft_coef, Size size) { // ��DFT
	Mat idft_coef, idft_img[2];
	shuffling(dft_coef, dft_coef);
	dft(dft_coef, idft_coef, DFT_INVERSE+DFT_SCALE);
	split(idft_coef, idft_img);

	Rect img_rect(Point(0, 0), size);
	idft_img[0](img_rect).convertTo(idft_img[0], CV_8U);
	return idft_img[0];
}

int main() {
	Mat image = imread("Lenna.png", 0);
	Rect img_rect(Point(0, 0), image.size());
	Mat dft_coef, dft_img, low_dft, high_dft, filtered_mat1, filtered_mat2;

	FFT(image, dft_coef, dft_img);  //FFT ����
	Mat low_filter = get_lowpassFilter(dft_coef.size(), 50);  // ������ ���� ����
	Mat high_filter = get_highpassFilter(dft_coef.size(), 20);  // ������ ���� ����

	multiply(dft_coef, low_filter, filtered_mat1);  //���͸�
	multiply(dft_coef, high_filter, filtered_mat2);
	log_mag(filtered_mat1, low_dft);  // ���ļ� ����Ʈ�� ����
	log_mag(filtered_mat2, high_dft);

	imshow("image", image);
	imshow("dft_img", dft_img);
	imshow("low_dft", low_dft);
	imshow("high_dft", high_dft);
	imshow("low_passed_img", IFFT(filtered_mat1, image.size()));
	imshow("high_passed_img", IFFT(filtered_mat2, image.size()));
	//imwrite("low_passed_img.jpg", IFFT(filtered_mat1, image.size()));
	//imwrite("high_passed_img.jpg", IFFT(filtered_mat2, image.size()));
	waitKey();
	return 0;
}