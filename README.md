# CPP_Computer_Vision_Studying

22.01 ~ 22.06 / hse@etri.re.kr

C++ 을 이용하여 전통적인 영상 처리 알고리즘을 직접 코딩하였습니다. 

Hough, SIFT 등 중요한 알고리즘을 이해하고, 구현하는 능력을 기를 수 있었습니다.

<br>

구체적인 구현 설명은 line-by-line 주석으로 명시하였습니다.


대표적인 알고리즘에 대한 목차는 다음과 같습니다.

## Table of Contents
- [Hough_Transform](#Hough_Transform)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Authors](#authors)

<br>

<hr>

# Hough_Transform
- 코드:
  [hough.cpp](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/blob/main/hough.cpp)

## Hough Transform(허프 변환) 이란?
허프 변환은 이미지에서 모양을 찾는 가장 유명한 방법입니다. 

이 방법을 이용하면 이미지의 형태를 찾거나, 누락되거나 깨진 영역을 복원할 수 있습니다.

이 코드에서는 허프 변환을 이용하여 __원하는 기울기__ 의 직선을 검출합니다.

<br>

기본적으로 허프변환의 직선의 방정식을 이용합니다. 

하나의 점을 지나는 무수한 직선의 방적식은 y=mx+c로 표현할 수 있으며, 

이것을 삼각함수를 이용하여 변형하면 r = 𝑥 cos 𝜃 + 𝑦 sin 𝜃 으로 표현할 수 있습니다.

<br>

#### 허프 변환에 대한 참고자료
- [ENG](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
- [KOR](https://opencv-python.readthedocs.io/en/latest/doc/25.imageHoughLineTransform/imageHoughLineTransform.html)

<br>

### Example
input image는 고속도로 이미지입니다. 우리는 허프 변환을 통해 이미지에서 고속도로의 실선 부분을 찾길 원합니다.

고속도로의 실선 부분은 직선의 기울기가 0\~50도, 120\~140도에 있습니다.

우리는 허프 변환을 이용하여 기울기가 0\~50도, 120\~140도인 직선을 검출하고, 그 중에서 가장 유력한 10개의 직선을 찾길 원합니다.

#### input Image

![highway](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/4a031ee1-3021-4dd4-96f4-27cdbf42a9e8)

우리의 허프 변환 코드를 실행시키면, 우선 아래의 사진과 같이 모든 직선을 검출할 수 있습니다.

#### All Lines

![all_the_lines](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/7ba6970c-8eb3-4ccb-9e68-926f03c04d0f)

이 중에서 직선의 기울기가 0\~50도, 120\~140도 이며, 가장 유력한 10개를 찾은 결과는 아래와 같습니다.

#### Top 10

![top10](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/9664cd15-5db5-4695-bd42-3b3094f3fd63)



