#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void hough_coord(Mat image, Mat& acc_mat, double rho, double theta) {
	int acc_h = (image.rows + image.cols) * 2 / rho;  //누적행렬 높이
	int acc_w = CV_PI / theta;  //누적행렬 너비
	acc_mat = Mat(acc_h, acc_w, CV_32S, Scalar(0));  //허프 누적행렬 선언

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			Point pt(x, y);
			if (image.at<uchar>(pt) > 0) { 
				/*for (int t = 0; t < acc_w; t++) {  //0~180도 반복
					double radian = t * theta;
					float r = pt.x * cos(radian) + pt.y * sin(radian);
					r = cvRound(r / rho + acc_mat.rows / 2);
					acc_mat.at<int>(r, t)++;
				}*/
				for (int t = 0; t < 51; t++) {  //0~50도
					double radian = t * theta;
					float r = pt.x * cos(radian) + pt.y * sin(radian);
					r = cvRound(r / rho + acc_mat.rows / 2);
					acc_mat.at<int>(r, t)++;
				}
				for (int t = 120; t < 141; t++) {  //120~140도
					double radian = t * theta;
					float r = pt.x * cos(radian) + pt.y * sin(radian);
					r = cvRound(r / rho + acc_mat.rows / 2);
					acc_mat.at<int>(r, t)++;
				}
			}
		}
	}
}

void acc_mask(Mat acc_mat, Mat& acc_dst, Size size, int thres) {
	acc_dst = Mat(acc_mat.size(), CV_32S, Scalar(0));
	Point h_m = size / 2;
	
	for (int r = h_m.y; r < acc_mat.rows - h_m.y; r++) {
		for (int t = h_m.x; t < acc_mat.cols - h_m.x; t++) {
			Point center = Point(t, r) - h_m;
			int c_value = acc_mat.at<int>(center);
			if (c_value >= thres) {  //50
				double maxVal = 0;
				for (int u = 0; u < size.height; u++) {
					for (int v = 0; v < size.width; v++) {
						Point start = center + Point(v, u);
						if (acc_mat.at<int>(start) >= maxVal) {
							maxVal = acc_mat.at<int>(start);
						}
					}
				}
				Rect rect(center, size);
				if (c_value >= maxVal) {
					acc_dst.at<int>(center) = c_value;
					acc_mat(rect).setTo(0);
				}
			}
		}
	}
}

void thres_lines(Mat acc_dst, Mat& lines, double _rho, double theta, int thres) {
	for (int r = 0; r < acc_dst.rows; r++) {
		for (int t = 0; t < acc_dst.cols; t++) {
			float value = (float)acc_dst.at<int>(r, t);
			if (value >= thres) {
				float rho = (float)((r - acc_dst.rows / 2) * _rho);
				float radian = (float)(t * theta);

				Matx13f line(rho, radian, value);
				lines.push_back((Mat)line);
			}
		}
	}
}

void sort_lines(Mat lines, vector<Vec2f>& s_lines) {
	Mat acc = lines.col(2), idx;
	sortIdx(acc, idx, SORT_EVERY_COLUMN + SORT_DESCENDING);

	for (int i = 0; i < idx.rows; i++) {
		int id = idx.at<int>(i);
		float rho = lines.at<float>(id, 0);
		float radian = lines.at<float>(id, 1);
		s_lines.push_back(Vec2f(rho, radian));
	}
}

void hough_lines(Mat src, vector<Vec2f>& s_lines, double rho, double theta, int thres) {
	Mat acc_mat, acc_dst, lines;
	hough_coord(src, acc_mat, rho, theta);
	acc_mask(acc_mat, acc_dst, Size(3, 7), thres);

	thres_lines(acc_dst, lines, rho, theta, thres);
	sort_lines(lines, s_lines);
}

void draw_houghLines_all(Mat image, Mat src, Mat& dst, vector<Vec2f> lines) {
	//cvtColor(src, dst, COLOR_GRAY2BGR);
	dst = Mat(image.size(), CV_8U, Scalar(0));
	cvtColor(dst, dst, COLOR_GRAY2BGR);
	for (size_t i = 0; i < (int)lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		Point2d pt(a * rho, b * rho);
		Point2d delta(1000 * -b, 1000 * a);
		line(dst, pt+delta, pt-delta, Scalar(255, 255, 255), 1, LINE_AA);
	}
}

void draw_houghLines(Mat image, Mat src, Mat& dst, vector<Vec2f> lines, int nline) {
	//cvtColor(src, dst, COLOR_GRAY2BGR);
	dst = Mat(image.size(), CV_8U, Scalar(0));
	cvtColor(dst, dst, COLOR_GRAY2BGR);
	for (size_t i = 0; i < min((int)lines.size(), nline); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		Point2d pt(a * rho, b * rho);
		Point2d delta(1000 * -b, 1000 * a);
		line(dst, pt + delta, pt - delta, Scalar(255, 255, 255), 1, LINE_AA);
	}
}

int main() {
	Mat image = imread("highway.jpg", IMREAD_GRAYSCALE);
	double rho = 1, theta = CV_PI / 180;  //rho:거리간격, theta:각도간격
	Mat canny;
	Mat dst1(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat dst2(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat want(image.size(), image.type(), Scalar(0));

	GaussianBlur(image, canny, Size(5, 5), 2, 2);
	Canny(canny, canny, 100, 150, 3);
	vector<Vec2f> lines;
	HoughLines(canny, lines, 1, CV_PI/180, 50); //canny, lines, 1, CV_PI, 180, 150

	for (size_t i = 0; i < (int)lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		cout << theta << endl;
		double a = cos(theta), b = sin(theta);
		Point2d pt(a * rho, b * rho);
		Point2d delta(1000 * -b, 1000 * a);
		line(dst1, pt + delta, pt - delta, Scalar(255, 253, 139), 1, LINE_AA);
	}

	hough_lines(canny, lines, rho, theta, 50);
	draw_houghLines_all(image, canny, dst1, lines);
	draw_houghLines(image, canny, dst2, lines, 10);
	/*Mat acc_mat, acc_dst;
	hough_coord(canny, acc_mat, rho, theta);
	acc_mask(acc_mat, acc_dst, Size(3, 7), 50);

	for (int i = 0; i < acc_dst.rows; i++) {
		for (int j = 0; j < acc_dst.cols; j++) {
			//cout << acc_dst.at<int>(i, j) << " ";
			if (acc_dst.at<int>(i, j) != 0) {
				double a = cos(theta), b = sin(theta);
				dst.at<uchar>(b*j, a*j) = 255;
			}
		}
		//cout << endl;
	}*/

	imshow("original", image);
	imshow("canny", canny);
	//imshow("want", want);
	imshow("all the lines", dst1);
	imshow("top10", dst2);
	imwrite("canny.jpg", canny);
	imwrite("all_the_lines.jpg", dst1);
	imwrite("top10.jpg", dst2);
	waitKey();
	return 0;
}