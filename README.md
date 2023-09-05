# CPP_Computer_Vision_Studying

22.01 ~ 22.06 

<br>

C++ 을 이용하여 전통적인 영상 처리 알고리즘을 직접 코딩하였습니다. 

Hough, RANSAC, SIFT 등 중요한 알고리즘을 이해하고, 구현하는 능력을 기를 수 있었습니다.

<br>

구체적인 구현 설명은 주석으로 명시하였습니다.

대표적인 알고리즘에 대한 목차는 다음과 같습니다.

<br>

## Table of Contents
- [RANSAC](#RANSAC)
- [Hough_Transform](#Hough_Transform)
- [Labeling_using_EQ_Table](#Labeling_using_EQ_Table)
- [Otsu_이진화](#Otsu_이진화)
- [DFT(Discrete Fourier Transform), 이산 푸리에 변환](#DFT)
- [Authors](#authors)

<br>

<hr>

# RANSAC
- 코드:
  [image_RANSAC.cpp](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/blob/main/image_RANSAC.cpp)

## RANSAC 이란?
최소자승법을 이용해 측정 오차나 노이즈를 제거하는 아주 유명한 알고리즘입니다.

<img width="694" alt="5" src="https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/51da324f-52bc-4ae9-beaa-a87ac4948696">

<br>
<br>

RANSAC은 다음과 같은 단계로 작동합니다:

```
1. 무작위로 최소한의 데이터 포인트를 선택하여 모델을 추정합니다.
2. 다른 모든 데이터 포인트에 대해 이 모델과의 거리를 계산합니다.
3. 일정 거리 이내의 데이터 포인트를 모델의 일관성 있는 데이터로 간주합니다.
4. 일관성 있는 데이터가 특정 임계값 이상이면 현재의 추정된 모델을 좋은 모델로 간주합니다.
5. 지정된 반복 횟수 동안 1~4 단계를 반복합니다.
6. 가장 많은 일관성 있는 데이터를 가진 모델을 최종 모델로 선택합니다.
```

우리는 이 RANSAC을 이용하여 segmentation된 이미지에서 원하는 영역만 가져오도록 노이즈를 제거하는 알고리즘을 제작했습니다.

<br>

이 알고리즘의 흐름은 다음과 같습니다.

```
1. Otsu 알고리즘을 이용해 자동 이진화
2. EQ Table을 이용한 Labeling 알고리즘을 이용해 Segmentation
3. 각 레이블에 대해 RANSAC 알고리즘을 적용해 직선 검출
4. 원하는 기울기의 직선을 가진 레이블만 남기고 나머지 제거
```

<br>

### Example
input image는 고속도로 이미지입니다. 우리는 RANSAC을 통해 이미지에서 고속도로의 실선 영역을 찾길 원합니다.

#### input Image

![highway](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/4a031ee1-3021-4dd4-96f4-27cdbf42a9e8)

먼저 Otsu 알고리즘을 이용해 자동으로 이진화 하면 다음의 이미지가 출력됩니다.

#### Otsu Threshold

![otsu](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/fbab7b9f-b61b-4f0f-bbad-6879ffb11ed7)

그 다음 EQ Table을 이용해 Labeling 한 뒤 각 레이블마다 다른 색을 칠하면 다음과 같습니다.

#### Labeling

![label_img](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/84135bf8-1e5e-4501-95a8-4662fe1f14ee)

마지막으로 RANSAC을 이용해 각 레이블에 대한 직선을 검출하고, 
원하는 직선의 기울기를 가진 레이블만 남겨놓습니다.

그 결과는 다음과 같습니다.

#### RANSAC

![ROI](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/da583e14-ee5e-450a-98aa-e5c720fb1cfa)


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

<br>

<hr>

# Labeling_using_EQ_Table
- 코드:
  [Labeling_using_EQ_Table.cpp](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/blob/main/Labeling_using_EQ_Table.cpp)

입력 이미지에서 자동적으로 segment들(나눠진 부분들)을 찾아 각각 다른 레이블을 붙이는 알고리즘입니다.

이 과정에서 우리는 EQ Table을 이용해 효율적으로 계산합니다.

#### EQ Table에 대한 참고자료
- https://hsyaloe.tistory.com/4

<br>

### Example
Input Image 에는 "영상처리 연습 한승은" 이라는 글자가 써있습니다.

#### input Image

![letter](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/9fe978db-3dd1-4f28-ad29-6295a05fc47d)

EQ Table을 이용해 한글의 각 자음 모음을 자동으로 segment하고, 

각각의 segment들에게 랜덤하게 다른 색상을 부여한 결과는 다음과 같습니다.

#### Output Image

![tmp](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/ca976fc9-8177-4141-b8d6-17fbb7490068)

<br>

<hr>

# Otsu_이진화
- 코드:
  [otsu.cpp](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/blob/main/otsu.cpp)

Otsu 알고리즘을 직접 구현하여 자동으로 이진화 threshold를 찾고, 이진화하는 알고리즘입니다.

<br>

### Example
0~255까지 밝기 구간 별로 나누어진 이미지가 들어갔을 때, 이진화 threshold를 자동으로 찾고 이진화 한 결과입니다.

#### input Image

![bright](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/4ab53fb5-6a64-485c-be40-a094e041dbbc)


#### Output Image

![otsu](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/d72a05df-1dc1-4d62-9590-6f0819827916)

<br>

# Otsu_삼진화
- 코드:
  [otsu_2thres.cpp](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/blob/main/otsu_2thres.cpp)

Otsu 알고리즘을 직접 구현하여 자동으로 2 개의 threshold를 찾고, 삼진화하는 알고리즘입니다.

<br>

### Example
0~255까지 밝기 구간 별로 나누어진 이미지가 들어갔을 때, 2 개의 threshold를 자동으로 찾고 삼진화 한 결과입니다.

#### input Image

![bright](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/4ab53fb5-6a64-485c-be40-a094e041dbbc)


#### Output Image

![result](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/b4e8670c-d42d-4753-a5e7-3b3f2570e825)

<br>

<hr>

# DFT
- 코드:
  [otsu.cpp](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/blob/main/otsu.cpp)

(Discrete Fourier Transform)_이산_푸리에_변환 

<br>

### Example
0~255까지 밝기 구간 별로 나누어진 이미지가 들어갔을 때, 이진화 threshold를 자동으로 찾고 이진화 한 결과입니다.

#### input Image

![bright](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/4ab53fb5-6a64-485c-be40-a094e041dbbc)


#### Output Image

![otsu](https://github.com/Seungeun-Han/CPP_Computer_Vision_Studying/assets/101082685/d72a05df-1dc1-4d62-9590-6f0819827916)

<br>



