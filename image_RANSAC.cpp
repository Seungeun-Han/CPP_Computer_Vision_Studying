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
	Mat image = imread("highway.jpg", IMREAD_GRAYSCALE);
	Mat otsu(image.size(), image.type(), Scalar(0));
	Mat result(image.size(), CV_32F, Scalar(0));
	Mat ROI(image.size(), CV_8UC3, Scalar(0, 0, 0));
	int thres = Calc_Thres_by_Otsu(image);

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) >= thres) {
				otsu.at<uchar>(i, j) = 255;
			}
		}
	}
	cout << "thresold: " << thres << endl;

	//erode(otsu, otsu, Mat::ones(Size(3, 3), CV_8UC1));
	//dilate(otsu, otsu, Mat::ones(Size(3, 3), CV_8UC1));
	//morphologyEx(otsu, otsu, MORPH_OPEN, Mat::ones(Size(5, 5), CV_8UC1));

	int label_count = 0;
	int* eq_table = new int[image.rows * image.cols + 1];
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (otsu.at<uchar>(i, j) > 0) {
				if (i == 0 && j == 0) {  //(0,0)
					label_count++;
					result.at<float>(i, j) = label_count;
					eq_table[label_count] = label_count;
					continue;
				}
				else if (i == 0 && j != 0) {  //0행
					if (otsu.at<uchar>(i, j - 1) > 0) {
						result.at<float>(i, j) = result.at<float>(i, j - 1);
						continue;
					}
					else {
						label_count++;
						result.at<float>(i, j) = label_count;
						eq_table[label_count] = label_count;
						continue;
					}
				}
				else if (i != 0 && j == 0) {  //0열
					if (otsu.at<uchar>(i - 1, j) > 0) {
						result.at<float>(i, j) = result.at<float>(i - 1, j);
						continue;
					}
					else {
						label_count++;
						result.at<float>(i, j) = label_count;
						eq_table[label_count] = label_count;
						continue;
					}
				}
				else if (j == image.cols - 1) {
					if ((i != 0 && j != 0) && otsu.at<uchar>(i - 1, j) == 0
						&& otsu.at<uchar>(i, j - 1) == 0 && otsu.at<uchar>(i - 1, j - 1) == 0) {
						label_count++;
						result.at<float>(i, j) = label_count;
						eq_table[label_count] = label_count;
					}
					else {  //그 외
						float values[3] = { 0, };
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

						float min = values[0];
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
					}
					continue;
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
					float values[4] = { 0, };
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

					float min = values[0];
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

	int l_count = 0;
	for (int i = 1; i <= label_count; i++) {
		if (i == eq_table[i]) {
			l_count++;
		}
	}
	cout << "The number of labels: " << l_count << endl;

	int n = 1;
	int* labels = new int[l_count + 1];
	for (int i = 1; i <= label_count; i++) {
		if (i == eq_table[i]) {
			labels[n] = i;
			n++;
		}
	}

	/*for (int i = 1; i <= l_count; i++) {
		cout << i << ": " << labels[i] << endl;
	}*/

	/*1부터 다시 labeling*/
	for (int k = 1; k <= l_count; k++) {
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				if (result.at<float>(i, j) == labels[k]) {
					result.at<float>(i, j) = (float)k;
				}
			}
		}
	}

	srand((unsigned int)time(NULL));
	Mat label_img(image.size(), CV_8UC3, Scalar(0, 0, 0));
	for (int k = 1; k <= l_count; k++) {
		Vec3b random_Color = Vec3b((int)rand() % 256, (int)rand() % 256, (int)rand() % 256);
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				if (result.at<float>(i, j) == (float)k) {
					label_img.at<Vec3b>(i, j) = random_Color;
				}
			}
		}
	}
	
	Mat dst, labels2, stats, centroids;
	int cnt = connectedComponentsWithStats(otsu, labels2, stats, centroids);
	cvtColor(image, dst, COLOR_GRAY2BGR);
	/*for (int i = 1; i < cnt; i++) {
		int* p = stats.ptr<int>(i);
		if (p[4] >= 100) {
			rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(255, 0, 255), 1);
			putText(dst, to_string(p[4]), Point(p[0], p[1]), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		}
	}*/

	int* area = new int[l_count + 1];
	for (int i = 1; i < l_count + 1; i++) {
		area[i] = 0;
	}

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (result.at<float>(i, j) != 0) {
				int label = (int)result.at<float>(i, j);
				area[label]++;
			}
		}
	}

	int iter = 5000; //반복 횟수  1000, 3000, 5000
	double t = 1.0;
	int d = 50; //인라이어 집합의 크기
	int e = 1; //직선 적합 오차
	for (int k = 1; k <= l_count; k++) {
		if (area[k] >= 300) {// 레이블 원소 개수가 300개 이상인 것만
			int* p = stats.ptr<int>(k);
			Rect rect(p[0], p[1], p[2], p[3]);
			Mat crop = otsu(rect);
			Mat canny;

			/*
			if (k == 4)
				imwrite("crop4.jpg", crop);
			else if (k == 16)
				imwrite("crop16.jpg", crop);
			else if (k == 180)
				imwrite("crop180.jpg", crop);*/


			Canny(crop, canny, thres, 150);

			int edge_count = 0;
			Vec2f* edge = new Vec2f[canny.rows * canny.cols];
			for (int i = 0; i < canny.rows; i++) {
				for (int j = 0; j < canny.cols; j++) {
					if (canny.at<uchar>(i, j) == 255) {
						edge[edge_count] = Vec2f(i, j);
						edge_count++;
					}
				}
			}

			if (edge_count == 0) {
				continue;
			}

			int max = 0;
			float line_info[2] = { 0, };
			srand((unsigned int)time(NULL));
			for (int loop = 1; loop < iter; loop++) {
				int p1 = (int)rand() % edge_count;
				int p2 = (int)rand() % edge_count;
				if (p1 == p2) {
					while (p1 != p2)
						p2 = (int)rand() % edge_count;
				}

				int inliner_count = 0;
				Vec2f* inliner = new Vec2f[edge_count];
				inliner[inliner_count] = edge[p1];
				inliner_count++;
				inliner[inliner_count] = edge[p2];
				inliner_count++;


				for (int i = 0; i < edge_count && i != p1 && i != p2; i++) {
					double distance = abs((edge[p2][0] - edge[p1][0]) * (edge[p1][1] - edge[i][1]) - (edge[p1][0] - edge[i][0]) * (edge[p2][1] - edge[p1][1]))
						/ sqrt(pow((edge[p2][0] - edge[p1][0]), 2) + pow((edge[p2][1] - edge[p1][1]), 2));
					if (distance <= t) {
						inliner[inliner_count] = edge[i];
						inliner_count++;
					}
				}
				
				if (inliner_count >= max) {  //집합 inliner가 max개 이상의 샘플을 가지면
					Mat A(inliner_count, 2, CV_32F, Scalar(1));
					for (int i = 0; i < inliner_count; i++) {
						A.at<float>(i, 0) = inliner[i][0];
					}
					Mat B(inliner_count, 1, CV_32F, Scalar(0));
					for (int i = 0; i < inliner_count; i++) {
						B.at<float>(i) = inliner[i][1];
					}

					Mat inv, mulA, A_t;
					transpose(A, A_t);
					mulA = A_t * A;
					invert(mulA, inv, DECOMP_SVD);

					Mat X(1, 2, CV_32F, Scalar(0));
					X = (inv * A_t) * B;
					line_info[0] = X.at<float>(0);
					line_info[1] = X.at<float>(1);
					max = inliner_count;
				}
				delete[] inliner;
			}

			float a = line_info[0];
			float b = line_info[1];
			cout << "\n";
			cout << "계산 결과" << endl;
			cout << "a: " << a << ", b: " << b << endl;
			cout << "따라서 label "<< k << " 에서 y = " << a << "x + (" << b << ") 직선 검출" << endl;

			if (a > -1 && a < 0.96) {
				//srand((unsigned int)time(NULL));
				//Vec3b random_Color = Vec3b((int)rand() % 256, (int)rand() % 256, (int)rand() % 256);
				for (int i = 0; i < image.rows; i++) {
					for (int j = 0; j < image.cols; j++) {
						if (result.at<float>(i, j) == k) {
							ROI.at<Vec3b>(i, j) = Vec3b(255,255,255); //dst.at<Vec3b>(i, j)
						}
					}
				}
			}
			delete[] edge;
		}
	}
	
	imshow("image", image);
	imshow("otsu", otsu);
	imshow("label_img", label_img);
	imshow("ROI", ROI);
	imwrite("otsu.jpg", otsu);
	imwrite("label_img.jpg", label_img);
	imwrite("ROI.jpg", ROI);
	waitKey();

	
	return 0;
}
