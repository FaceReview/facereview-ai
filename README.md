# Facial Expression Recognition
한국인의 표정으로 학습 된 AI 모델을 설계한다.
## Summary
1. AIHub에서 지원하는 [한국인 표정 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82)을 활용했다.
2. ResNet50, ResNet152 모델을 사용하여 프로젝트에 더 적합한 모델을 찾는다.
3. 학습에 필요한 이미지 수를 조정하여 각 표정 별 정확도 편차가 최소화된 모델을 구했다.
4. 학습에 사용되지 않은 30,000장의 테스트 데이터셋으로 약 89%의 정확도를 얻었다.
5. **public : private = 1 : 1** 비율의 표정 이미지 100장으로 설문조사하여 **사람 vs 인공지능모델**의 성능을 비교하였다.
## 1. Dataset
AIHub에서 지원하는 한국인 표정 데이터셋 중 식별하기 어렵고 애매하다고 판단한 불안, 상처 감정을 제외한 5개 (기쁨, 당황, 분노, 슬픔, 중립) 감정의 이미지를 학습 데이터셋으로 활용하였습니다.

학습 이미지 전처리를 위해 cvlib 의 `detect_face` 함수를 사용하여 face crop 이미지를 얻은 후 96x96 으로 resize 한 후 Grayscale 을 진행하였습니다. 전처리를 위해 사용한 코드는 [preprocessing.py]() 에서 확인할 수 있습니다.

생성된 이미지는 약 300,000장이고 이 중 학습에 사용하지 않을 데이터 30,000장을 선별하여 테스트 데이터셋으로 사용하였습니다.
![dataset preview]()

## 2. 학습 모델 ResNet50
ResNet은 마이크로소프트에서 개발하여 2015년 [ILSVRC (ImageNet Large Scale Visual Recognition Challenge)](https://image-net.org/challenges/LSVRC/) 에서 우승을 차지한 알고리즘입니다.

기존 CNN 이미지 인식 분야에서 모델의 성능 향상을 위해 layer 를 깊게 쌓는 방식을 채택하였는데, 실제로는 layer 가 20층 이상일 때 성능이 낮아지는 문제가 발견되었습니다. 이를 해결하기 위해 Residual Learning 을 사용하여 설계한 네트워크가 ResNet 입니다. 모델에 대한 자세한 설명은 [여기](https://github.com/FaceReview/facereview-ai/blob/master/docs/ResNET.pdf) 있습니다.

ResNet 네트워크를 50층으로 설계한 것이 ResNet50, 152층으로 설계된 것이 ResNet152 입니다. 두 네트워크를 모두 구현하고 비교하여 최종 네트워크를 선택한 과정은 [여기]() 있습니다.

학습을 위해 사용한 코드는 [resnet50.py]() 에서 볼 수 있습니다.

## 3. 성능
### 3.1 정확도
학습된 여러 모델의 정확도를 측정하고 비교하기 위해서 학습에 관여하지 않은 30,000장의 이미지를 사용하였습니다.

각 이미지는 `preprocessing.py` 를 실행하여 얻어진 이미지이며, 각 표정 별 5,000장씩 동일하게 존재하여 표정 별 정확도를 알 수 있습니다.

여러 모델을 비교하며 최종 모델을 얻기까지의 과정은 [여기](https://github.com/FaceReview/facereview-ai/blob/master/docs/performance.md) 에서 확인할 수 있습니다.

추가로, 표정 별 정확도의 편차를 줄이기 위한 노력은 [여기]() 에서 확인할 수 있습니다.

![best performance chart]()

### 3.2 설문조사
최종 모델의 성능을 파악하기 위해 설문조사 방식을 채택하였습니다.

**public : private = 1 : 1** 로 구성된 100장의 이미지로 총 40명에게 설문조사하였습니다.  
public은 학습에 관여하지 않은 AIHub 데이터셋이고 private은 프로젝트 진행자 및 지인들 등의 이미지로 구성된 데이터셋입니다.

설문조사 결과 설문자들은 평균 약 79개의 표정을 인식하였습니다.

같은 데이터셋에 대하여 AI 모델은 78개의 표정을 인식하였습니다.

셀문조사에 대한 자세한 결과는 [여기](https://github.com/FaceReview/facereview-ai/blob/master/docs/survey.md) 에서 확인할 수 있습니다.


![survey](https://github.com/FaceReview/facereview-ai/blob/master/img/google_form.PNG)

![AI](https://github.com/FaceReview/facereview-ai/blob/master/img/AI.PNG)

## reference
https://github.com/kitae0522/Facial-Expression-Recognition