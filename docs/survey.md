# 모델 성능 측정을 위한 설문조사
- **public : private = 1 : 1** 로 구성된 100장의 이미지로 총 42명에게 설문조사하였습니다.  
public은 학습에 관여하지 않은 AIHub 데이터셋이고 private은 프로젝트 진행자 및 지인들 등의 이미지로 구성된 데이터셋입니다.
- private 데이터는 프로젝트 진행자들이 라벨링 하였습니다.
- 데이터셋은 각 표정 별 20장의 사진으로 구성되어 있어서 AI 예측에 대한 표정 별 정확도를 구할 수 있습니다.
- AI 학습 환경과 동일한 환경에서 설문을 하기 위하여 각 이미지에 대하여 `preprocessing.py` 를 거친 이미지를 사용하였습니다.

---
## 설문조사 결과
![google_form](https://github.com/FaceReview/facereview-ai/blob/master/img/google_form.PNG)

설문자들은 평균 약 79개의 이미지를 인식하였습니다.

## AI 예측 결과
![AI](https://github.com/FaceReview/facereview-ai/blob/master/img/AI.PNG)


AI 는 78개의 이미지를 인식하였습니다.

---
## Human vs AI
다음은 AI 와 설문조사의 결과를 비교한 것입니다.

![1](https://github.com/FaceReview/facereview-ai/blob/master/img/1.PNG)
  

![2](https://github.com/FaceReview/facereview-ai/blob/master/img/2.PNG)
  

![3](https://github.com/FaceReview/facereview-ai/blob/master/img/3.PNG)
  

![4](https://github.com/FaceReview/facereview-ai/blob/master/img/4.PNG)
  

![5](https://github.com/FaceReview/facereview-ai/blob/master/img/5.PNG)
  

![6](https://github.com/FaceReview/facereview-ai/blob/master/img/6.PNG)
  

![7](https://github.com/FaceReview/facereview-ai/blob/master/img/7.PNG)
