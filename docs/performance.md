# Model Performance
모델 성능 측정을 위해서 학습에 사용되지 않은 이미지 각 표정 별 6,000장(총 30,000장)을 사용하였음.

## 1. model 1.x - 고정 epoch 수로 학습
#### model 1.0
- 각 표정 별 1,000장의 이미지 (총 5,000장)
- epoch = 20
- batch_size = 8
#### model 1.1
- 각 표정 별 5,000장의 이미지 (총 25,000장)
- epoch = 20
- batch_size = 8
#### model 1.2
- 각 표정 별 10,000장의 이미지 (총 50,000장)
- epoch = 20
- batch_size = 8
#### model 1.3
- 각 표정 별 50,000장의 이미지 (총 25,000장)
- epoch = 20
- batch_size = 16

|accuracy|model 1.0|model 1.1|model 1.2|model 1.3|
|:-:|:-:|:-:|:-:|:-:|
|happy|||||
|surprise|||||
|angry|||||
|sad|||||
|neutral|||||
|total|||||
|epochs|||||
|학습속도(s)|||||
---

## 2. model 2.x - EarlyStopping 함수 사용 : monitor (val_loss)
#### model 2.0
- 각 표정 별 1,000장의 이미지 (총 5,000장)
- batch_size = 8
#### model 2.1
- 각 표정 별 5,000장의 이미지 (총 25,000장)
- batch_size = 8
#### model 2.2
- 각 표정 별 10,000장의 이미지 (총 50,000장)
- batch_size = 8
#### model 2.3
- 각 표정 별 50,000장의 이미지 (총 25,000장)
- batch_size = 16

|accuracy|model 2.0|model 2.1|model 2.2|model 2.3|
|:-:|:-:|:-:|:-:|:-:|
|happy|||||
|surprise|||||
|angry|||||
|sad|||||
|neutral|||||
|total|||||
|epochs|||||
|학습속도(s)|||||
---

## 3. model 3.x - EarlyStopping 함수 사용 : monitor (val_accuracy)
### model 3.0
- 각 표정 별 1,000장의 이미지 (총 5,000장)
- batch_size = 8

### model 3.1
- 각 표정 별 5,000장의 이미지 (총 25,000장)
- batch_size = 8

### model 3.2
- 각 표정 별 10,000장의 이미지 (총 50,000장)
- batch_size = 8

### model 3.3
- 각 표정 별 50,000장의 이미지 (총 250,000장)
- batch_size = 16

|accuracy|model 3.0|model 3.1|model 3.2|model 3.3|
|:-:|:-:|:-:|:-:|:-:|
|happy|83.48|89.15|89.46|95.71|
|surprise|64.01|70.75|80.80|85.68|
|angry|62|62.01|60.88|80.81|
|sad|48.4|74.65|73.13|88.25|
|neutral|65.46|77.85|74.08|92.23|
|total|64.67|74.88|75.67|88.54|
|epochs|116|55|38|83|
|학습속도(s)|7544|18214|25001|241983|

