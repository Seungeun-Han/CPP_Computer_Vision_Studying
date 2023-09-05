#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void filter(Mat image, Mat& result, Mat mask) {
	result = Mat(image.size(), CV_32F, Scalar(0));
	Point h_m = mask.size() / 2;
	for (int i = h_m.y; i < image.rows - h_m.y; i++) {
		for (int j = h_m.x; j < image.cols - h_m.x; j++) {
			float sum = 0;
			for (int u = 0; u < mask.rows; u++) {
				for (int v = 0; v < mask.cols; v++) {
					int y = i + u - h_m.y;
					int x = j + v - h_m.x;
					sum += image.at<uchar>(y, x) * mask.at<float>(u, v);
				}
			}
			result.at<float>(i, j) = sum;
		}
	}
}

void differential(Mat image, Mat& dst, float data1[], float data2[], int x, int y) {
	Mat mask;
	if (x == 1 && y == 0) {
		mask = Mat(3, 3, CV_32F, data1);
	}
	else if(x == 0 && y == 1){
		mask = Mat(3, 3, CV_32F, data2);
	}

	filter(image, dst, mask);
	//dst = abs(dst);
}

void cornerharris(Mat image, Mat& corner, int bsize, int ksize, double k) {
	float sobel_x[] = {
		-1,0,1,
		-2,0,2,
		-1,0,1

	};
	float sobel_y[] = {
		-1,-2,-1,
		0,0,0,
		1,2,1
	};

	/*float GaussianMask[] = {
		0.0751, 0.1238, 0.0751,
		0.1238, 0.2042, 0.1238,
		0.0751, 0.1238, 0.0751
	};*/

	Mat dx, dy, dxy, dxx, dyy, hor_sobel, ver_sobel;
	corner = Mat(image.size(), CV_32F, Scalar(0));
	
	differential(image, dx, sobel_x, sobel_y, 1, 0);
	differential(image, dy, sobel_x, sobel_y, 0, 1);

	//Sobel(image, dx, CV_32F, 1, 0, ksize);  
	//Sobel(image, dy, CV_32F, 0, 1, ksize);
	multiply(dx, dx, dxx);  //행렬 원소간 곱 계산 -> 미분 행렬 제곱
	multiply(dy, dy, dyy);
	multiply(dx, dy, dxy);

	Size msize(5, 5);
	GaussianBlur(dxx, dxx, msize, 0);
	GaussianBlur(dyy, dyy, msize, 0);
	GaussianBlur(dxy, dxy, msize, 0);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			float a = dxx.at<float>(i, j);
			float b = dyy.at<float>(i, j);
			float c = dxy.at<float>(i, j);
			corner.at<float>(i, j) = (a * b - c * c) - k * (a + b) * (a + b);
		}
	}
}

void draw_corner(Mat corner, Mat& image, float thresh) {
	int cnt = 0;
	normalize(corner, corner, 0, 100, NORM_MINMAX, CV_32F, Mat());

	for (int i = 1; i < corner.rows-1; i++) {  //비최대 억제
		for (int j = 1; j < corner.cols-1; j++) {
			float cur = corner.at<float>(i, j);
			
			if (cur > thresh) {
				if (cur > corner.at<float>(i - 1, j) &&
					cur > corner.at<float>(i + 1, j) &&
					cur > corner.at<float>(i, j - 1) &&
					cur > corner.at<float>(i, j + 1)) {
					circle(image, Point(j, i), 4, Scalar(0, 0, 255), -1);
					cnt++;
				}
				/*circle(image, Point(j, i), 4, Scalar(0, 0, 255), -1);
				cnt++;*/
			}
		}
	}
	cout << "특징점 개수: " << cnt << endl;
}

int main() {
	Mat image = imread("fig.jpg", IMREAD_COLOR);  //figures.jpg  //figures2.jpg 
	Mat dst(image.size(), image.type(), Scalar(0));
	image.copyTo(dst);
	int blocksize = 4; //이웃화소 범위
	int apertureSize = 3; //소벨 마스크 크기
	double k = 0.04; //적절한 것으로 알려짐
	float thresh = 30.0; //코너 임계값
	Mat gray, corner;

	cvtColor(image, gray, COLOR_BGR2GRAY);
	cornerharris(gray, corner, blocksize, apertureSize, k);
	draw_corner(corner, dst, thresh);

	imshow("original", image);
	imshow("gray", gray);
	imshow("dst", dst);
	waitKey();
	return 0;
}