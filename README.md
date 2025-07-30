본 실험에서는 고양이, 개, 호랑이, 하이에나 네 종류의 동물 얼굴을 대상으로 심층망을 설계하고, 이를 통해 최적으로 동물 얼굴을 인식하는 모델을 구현하고자 한다. 본 실험의 주요 목표는 데이터 증강 기법을 통해 학습 데이터를 강화하고, 최적의 신경망 구조를 c와 python으로 구성하고, 여러 하이퍼파라미터 설정을 찾아가는 비교 실험을 통해 동물 얼굴 인식 정확도를 극대화하는 것이다. 

이를 위해 이미지의 크기 조정, 회전, 반전, 노이즈 추가 등 다양한 데이터 증강 기법을 적용하여 데이터셋을 구성하고, 모델의 일반화 능력을 향상시키고자 하였다. 데이터 증강을 통한 데이터의 다양성 확보, 최적의 신경망 구조 설계 및 하이퍼파라미터 튜닝을 통해 동물의 얼굴을 효과적으로 인식할 수 있는 신경망을 구축하고자 한다


---

# 데이터셋 만들기

처음 신경망을 구현하기에 앞서서 데이터셋을 만들고자 하였고, 고양이, 개, 호랑이, 하이에나 사진을 얼굴만 크롭한 것으로 각각 30장을 준비하고, 256x256크기로 맞춘 후 흑백으로 변환하였습니다.

그 뒤에 그 사진들을 파이썬 스크립트를 사용하여 크롭한 얼굴을 왼쪽과 오른쪽으로 15도 회전시킨 것과, 좌우 반전 시켜서 이미지 데이터셋을 증강하였습니다.

또한 평가용 데이터셋 이미지들에는 과제 조건에 노이즈를 첨가하여 생성하도록 하였습니다. 

아래는 데이터셋 증강코드입니다.

```python
import os
from PIL import Image, ImageOps
import numpy as np
import random

# 이미지 저장 디렉토리 생성
os.makedirs("augmented_images/train", exist_ok=True)
os.makedirs("augmented_images/test", exist_ok=True)

# 학습용 이미지 증강 및 저장 함수 (노이즈 제외)
def augment_and_save_train_image(image, label_name, img_count, save_dir):
    angles = [-15, 0, 15]
    for angle in angles:
        rotated_image = image.rotate(angle)
        rotated_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_rot{angle}.png"))
        img_count += 1
        
        # 좌우 반전
        mirrored_image = ImageOps.mirror(rotated_image)
        mirrored_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_mirrored_rot{angle}.png"))
        img_count += 1

    return img_count

# 평가용 이미지 증강 및 저장 함수 (노이즈 포함)
def augment_and_save_test_image(image, label_name, img_count, save_dir):
    angles = [-15, 0, 15]
    for angle in angles:
        rotated_image = image.rotate(angle)
        rotated_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_rot{angle}.png"))
        img_count += 1
        
        # 좌우 반전
        mirrored_image = ImageOps.mirror(rotated_image)
        mirrored_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_mirrored_rot{angle}.png"))
        img_count += 1
    
    # 노이즈 추가 및 저장
    noisy_array = np.array(image) / 255.0 + (np.random.randn(*image.size) * 0.05)
    noisy_image = Image.fromarray(np.uint8(np.clip(noisy_array * 255, 0, 255)))
    noisy_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_noisy.png"))
    img_count += 1

    return img_count

# 데이터셋 디렉토리와 클래스 설정
data_dir = "C:/Users/최승렬/Desktop/AI/dataset"
classes = ["cat", "dog", "tiger", "hyena"]
train_ratio = 0.8  # 학습 데이터 비율

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    img_files = os.listdir(class_dir)
    train_size = int(len(img_files) * train_ratio)
    
    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(class_dir, img_name)
        image = Image.open(img_path).convert("L").resize((256, 256))  # 흑백 변환 및 크기 조정
        
        # 학습용 데이터 저장 (노이즈 제외)
        if idx < train_size:
            img_count = augment_and_save_train_image(image, class_name, idx, "augmented_images/train")
        # 평가용 데이터 저장 (노이즈 포함)
        else:
            img_count = augment_and_save_test_image(image, class_name, idx, "augmented_images/test")

print("증강된 이미지 파일들이 저장되었습니다.")

```

이로 인해 각 이미지가 회전과 대칭에 대한 정보를 포함했지만 이미지 파일 이름이 클래스 정보를 포함하고 있으므로, 파일 이름을 파싱하여 각 이미지가 해당하는 클래스에 맞게 라벨을 지정하는 함수를 추가로 작성하였습니다.

그리고 데이터셋의 양에 따른 성능의 차이를 확인하고자 하여 데이터셋 증강을 수행할 수 있는 코드를 작성하여 데이터셋의 이미지 양을 늘려보았습니다.
아래는 데이터셋 증강 코드입니다.

```jsx
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import random

# 이미지 저장 디렉토리 생성
os.makedirs("augmented_images/train2", exist_ok=True)
os.makedirs("augmented_images/test2", exist_ok=True)

# 학습용 이미지 증강 및 저장 함수
def augment_and_save_train_image(image, label_name, img_count, save_dir):
    # 회전 각도 설정
    angles = [-30, -15, 0, 15, 30]
    for angle in angles:
        rotated_image = image.rotate(angle)
        
        # 확대/축소 (비율 ±10%~20%)
        scale_factors = [0.9, 1.0, 1.1]
        for scale in scale_factors:
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            scaled_image = rotated_image.resize(new_size).resize(image.size)
            
            # 밝기 및 대비 조정
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            enhanced_image = ImageEnhance.Brightness(scaled_image).enhance(brightness_factor)
            enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast_factor)
            
            # 저장
            enhanced_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_rot{angle}_scale{scale}.png"))
            img_count += 1
            
            # 좌우 반전
            mirrored_image = ImageOps.mirror(enhanced_image)
            mirrored_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_mirrored_rot{angle}_scale{scale}.png"))
            img_count += 1

    return img_count

# 평가용 이미지 증강 및 저장 함수 (노이즈 추가 포함)
def augment_and_save_test_image(image, label_name, img_count, save_dir):
    angles = [-30, -15, 0, 15, 30]
    for angle in angles:
        rotated_image = image.rotate(angle)
        
        # 확대/축소
        scale_factors = [0.9, 1.0, 1.1]
        for scale in scale_factors:
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            scaled_image = rotated_image.resize(new_size).resize(image.size)
            
            # 밝기 및 대비 조정
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            enhanced_image = ImageEnhance.Brightness(scaled_image).enhance(brightness_factor)
            enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast_factor)
            
            # 저장
            enhanced_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_rot{angle}_scale{scale}.png"))
            img_count += 1
            
            # 좌우 반전
            mirrored_image = ImageOps.mirror(enhanced_image)
            mirrored_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_mirrored_rot{angle}_scale{scale}.png"))
            img_count += 1
    
    # 노이즈 추가
    noisy_array = np.array(image) / 255.0 + (np.random.randn(*image.size) * 0.05)
    noisy_image = Image.fromarray(np.uint8(np.clip(noisy_array * 255, 0, 255)))
    noisy_image.save(os.path.join(save_dir, f"{label_name}_{img_count}_noisy.png"))
    img_count += 1

    return img_count

# 데이터셋 디렉토리와 클래스 설정
data_dir = "C:/Users/최승렬/Desktop/AI/dataset"
classes = ["cat", "dog", "tiger", "hyena"]
train_ratio = 0.8  # 학습 데이터 비율

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    img_files = os.listdir(class_dir)
    train_size = int(len(img_files) * train_ratio)
    
    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(class_dir, img_name)
        image = Image.open(img_path).convert("L").resize((256, 256))  # 흑백 변환 및 크기 조정
        
        # 학습용 데이터 저장
        if idx < train_size:
            img_count = augment_and_save_train_image(image, class_name, idx, "augmented_images/train2")
        # 평가용 데이터 저장
        else:
            img_count = augment_and_save_test_image(image, class_name, idx, "augmented_images/test2")

print("증강된 이미지 파일들이 저장되었습니다.")
```

추가된 부분은 회전 각도를 ±30° 범위로 확대하고, ±10%~20% 비율로 확대/축소를 적용하였습니다. 또한 각 이미지에 랜덤으로 밝기 및 대비를 조정하여 적용하였습니다.

총 데이터셋은 train 데이터셋을 576장에서 2880장으로 늘렸고, test 데이터셋은 168장에서 744장으로 증강시켰습니다. 실험은 이 두가지의 데이터셋으로 진행하였습니다. 데이터셋의 호칭을 호칭을 576장을 데이터셋1, 2880장을 데이터셋2으로 지정하고 진행하겠습니다.

### 데이터셋 이미지 파일 형식

이미지 파일 형식을 이미지 파일을 직접 읽어 픽셀 데이터를 C 언어로 처리하고자 간단한 그레이스케일 포맷인 pgm으로 이미지를 변환하였습니다. 이를 선택한 이유는 파일 구조가 단순해, 파일 헤더를 읽어와 픽셀 데이터에 접근하는 방식이기에 pgm형식을 선택하였습니다.

아래는 pgm형식으로 전환하는 코드입니다.

```jsx
import os
from PIL import Image

def convert_all_png_to_pgm(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 입력 디렉토리 내 모든 파일 탐색
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".png"):
            png_path = os.path.join(input_directory, filename)
            pgm_filename = os.path.splitext(filename)[0] + ".pgm"
            pgm_path = os.path.join(output_directory, pgm_filename)

            # 이미지 열기 및 변환
            with Image.open(png_path) as img:
                # Grayscale로 변환
                img = img.convert("L")
                img.save(pgm_path, format="PPM")
            
            print(f"Converted {png_path} to {pgm_path}")

input_directory = "C:/Users/최승렬\Desktop/AI/augmented_images/train"        
output_directory = "C:/Users/최승렬/Desktop/AI/augmented_images/train_pgm"  
convert_all_png_to_pgm(input_directory, output_directory)

```


# python 버전 신경망

## 라이브러리 임포트

```python
import numpy as np
import os
import matplotlib.pyplot as plt
```

- 배열 및 행렬 연산을 사용하고자 수치 연산을 위한 numpy라이브러리를 사용하였습니다.
- 운영체제와 상호작용하기 위한 모듈로, 파일 및 디렉토리 경로 조작에 사용되는 os라이브러리를 사용했습니다
- 그래프와 시각화를 위해 matplotlib.pyplot 라이브러리를 사용했습니다.

---

## 하이퍼파라미터

```python
NUM_CLASSES = 4
BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.0001
EPOCHS = 10
INPUT_NODES = 256 * 256
HIDDEN_LAYERS = 4
HIDDEN_NODES = [32,16,16,8]
OUTPUT_NODES = NUM_CLASSES
L2_LAMBDA = 0.005
EPSILON = 1e-5
MAX_GRAD_NORM = 5.0
animal_names = ["cat", "dog", "tiger", "hyena"]
```

- **NUM_CLASSES**: 분류할 객체의 수로, 4개의 동물 클래스입니다.
- **BATCH_SIZE**: 한 번의 학습 단계에서 사용할 데이터 샘플의 수로 32로 설정했습니다.
- **INITIAL_LEARNING_RATE**: 초기 학습률입니다.
- **EPOCHS**: 전체 데이터셋을 몇 번 반복하여 학습할 것인지 결정하는 에폭입니다.
- **INPUT_NODES**: 입력층의 노드 수로, 신경망의 입력층에 전달되는 데이터의 크기, 이미지의 픽셀 수입니다.
- **HIDDEN_LAYERS**: 은닉층의 수입니다.
- **HIDDEN_NODES**: 각 은닉층의 노드 수를 리스트로 정의하였습니다.
- **OUTPUT_NODES**: 출력층의 노드 수로 4개의 클래스 중 하나로 분류해야 하기에 다음과 같이 설정하였다.
- **L2_LAMBDA**: L2 정규화의 람다 값으로, 과적합을 방지하기 위해 사용하였습니다.
- **EPSILON**: 작은 값으로, 계산 중에 0으로 나누는 오류를 방지하고자 사용하였습니다.
- **MAX_GRAD_NORM**: 그레디언트 클리핑을 위한 임계값으로, 그레디언트 폭주를 방지하고자 사용하였습니다.
- **animal_names**: 클래스 라벨을 정의한 리스트입니다.

실험을 진행할 때는 BATCH_SIZE, INITIAL_LEARNING_RATE, HIDDEN_LAYERS, HIDDEN_NODES, L2_LAMBDA, MAX_GRAD_NORM의 값들을 변형시켜가며 진행하였습니다.

---

## 데이터셋 로드 함수

### load_pgm_image

```python
def load_pgm_image(file_path):
    with open(file_path, 'rb') as file:
        file.readline()  # "P5"
        file.readline()  # width height
        file.readline()  # max_val
        data = np.fromfile(file, dtype=np.uint8)
        data = data.reshape((256, 256)).astype(np.float32) / 255.0
    return data.flatten()
```

- 데이터셋의 이미지들이 pgm형식으로 되어있는데, pgm 이미지 파일을 읽어와서 일차원 배열로 반환하는 함수입니다**.**
- 바이너리 모드로 파일을 열고, 헤더 부분("P5", 이미지 크기, 최대 값)을 읽어 넘깁니다.
- 픽셀 데이터를 np.uint8 타입으로 읽어옵니다.
- 이미지를 (256, 256) 형태로 재구성하고, float32로 변환 후 0~1 사이로 정규화합니다.
- 이미지를 평탄화하여 일차원 배열로 반환합니다.

### get_label_from_filename

```python
def get_label_from_filename(filename):
    filename = filename.lower()
    for i, name in enumerate(animal_names):
        if name in filename:
            return i
    print(f"Warning: Label not found for file {filename}")
    return 1  # 라벨을 찾지 못한 경우
```

- 파일명에서 클래스 라벨을 추출하는 함수입니다.
- 데이터셋 증강을 위해 이미지들을 조정하다보니 이미지의 이름들에서 class 말고 cat_24_rot-15 이런식으로 되어있기에 클래스 라벨을 추출할 필요성을 느껴서 추가하였습니다.
- 이미지들 이름에서 클래스들이 전부 소문자로 되어있기에 파일명을 소문자로 변환하여 일관성을 유지하고자 하였습니다.
- animal_names 리스트를 순회하며 파일명에 해당 동물명이 포함되어 있는지 확인합니다.
- 해당 동물명이 포함되어 있으면 그 인덱스를 라벨로 반환합니다.
- 해당되지 않는 경우 -1을 반환하고 경고 메시지를 출력합니다.

### load_images_with_labels

```python
def load_images_with_labels(folder_path, filenames):
    data = []
    labels = []
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        data.append(load_pgm_image(file_path))
        label = get_label_from_filename(filename)
        if label == -1:
            print(f"Warning: Label not found for file {filename}")
        labels.append(label)
    return np.array(data), np.array(labels)
```

- 지정된 폴더에서 파일명을 받아 이미지와 라벨을 로드하는 함수입니다.
- 각 이미지 파일에 대해 load_pgm_image를 호출하여 데이터를 읽어옵니다.
- 파일명에서 라벨을 추출하기 위해 get_label_from_filename 함수를 사용합니다.
- 데이터를 NumPy 배열로 변환하여 반환합니다.

---

## 활성화 함수

### leaky_relu

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

- Leaky ReLU 활성화 함수를 구현하였습니다.
- 입력 x가 양수이면 그대로 반환하고, 음수이면 alpha를 곱하여 기울기를 유지합니다.
- alpha는 음수 영역의 기울기로 기본값은 0.01로 설정하였습니다.

활성화 함수는 신경망의 각 뉴런에서 입력을 처리해 출력으로 변환하는 비선형 함수로 신경망의 은닉층에서 입력 데이터에 비선형성을 추가하고 안정적으로 학습할 수 있도록 돕는 중요한 구성 요소이기에 처음에 무엇을 선택할지 고민이 많았습니다. 

기존에는 Relu로 실험을 진행하였다가 죽은 ReLU 문제, 즉 입력이 0 이하일 때 출력이 0이 되어 이로 인해 뉴런이 비활성화되어 가중치가 업데이트되지 않는 현상이 발생하는지 확인하였고, 실제로 그러하여서 leaky_relu로 바꾸어서 음수 입력에 대해 작은 기울기 alpha를 적용하여 뉴런이 완전히 죽지 않도록 진행하였습니다.


### leaky_relu_derivative

```python
def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx
```

- Leaky ReLU의 도함수를 계산하는 함수로 ****역전파 시 그레디언트 계산에 사용됩니다.
- 입력 x가 양수이면 도함수는 1, 음수이면 alpha입니다.

---

## He 초기화 함수

```python
def he_initialize(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2.0 / rows)
```

- 가중치를 초기화 하는 함수로 He 초기화 함수를 선택하였습니다. 깊은 신경망에서의 기울기 소실 문제를 완화할 수 있는 역할을 합니다.
- 표준 정규분포에서 난수를 생성하고, 분산을 2.0 / rows로 조정합니다.
- rows는 입력 노드 수, cols는 출력 노드 수입니다.

신경망의 학습 안정성과 효율성을 보장하는 적절한 가중치 초기화와 편향 초기화 선정에 고민이 많았았습니다.

xavier 초기화 함수로도 실험을 진행했었으나 he 초기화가 Relu계열 활성화 수에 최적화 되있기에 he초기화를 선정하였고, 신경망의 깊이가 깊어질 때 그레디언트 폭주나 소실 문제를 어느 정도 완화할 수 있기에 초기화 함수를 He 초기화 함수로 선정하고 진행하였습니다.

---

## 네트워크 초기화 함수

```python
def init_network():
    weights = []
    biases = []

    for i in range(HIDDEN_LAYERS):
        in_nodes = INPUT_NODES if i == 0 else HIDDEN_NODES[i - 1]
        out_nodes = HIDDEN_NODES[i]
        weights.append(he_initialize(in_nodes, out_nodes))
        biases.append(np.zeros((1, out_nodes)))
       

    # 출력층
    weights.append(he_initialize(HIDDEN_NODES[-1], OUTPUT_NODES))
    biases.append(np.zeros((1, OUTPUT_NODES)))
    return weights, biases
```

- 신경망의 가중치와 편향을 초기화하는 함수입니다.
- 은닉층의 수만큼 루프를 돌며 각 층의 가중치와 편향을 초기화합니다.
- 첫 번째 층의 입력 노드 수는 INPUT_NODES이며, 그 이후 층은 이전 층의 노드 수를 사용합니다.
- 가중치는 he_initialize 함수를 사용하여 초기화하고, 편향은 0으로 초기화합니다.
- 출력층의 가중치와 편향도 동일하게 초기화합니다.

---

## 순전파 함수

```python
def forward(weights, biases, input_data):
    caches = []
    out = input_data

    for i in range(HIDDEN_LAYERS):
        z = out @ weights[i] + biases[i]
        a = leaky_relu(z)
        caches.append((out, weights[i], biases[i], z))
        out = a

    # 출력층
    z = out @ weights[-1] + biases[-1]
    z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    softmax_output = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    caches.append((out, weights[-1], biases[-1], None))
    return softmax_output, caches
```

- 순전파를 수행하여 입력 데이터로부터 예측값을 계산하는 함수입니다.
- caches 리스트는 역전파 시 필요한 중간 계산 값을 저장합니다.{효율적인 학습(역전파)을 위해}
- 각 은닉층에 대해:
    - z는 선형 변환 결과이며, z = out @ weights[i] + biases[i]로 계산됩니다.
    - a는 활성화 함수를 적용한 결과입니다.
    - caches에 입력, 가중치, 편향, z를 저장합니다.
    - out은 다음 층의 입력이 됩니다.
- 출력층에서:
    - 선형 변환 후 softmax 함수를 적용하여 클래스 확률을 계산합니다.
    - 안정성을 위해 z에서 최대값을 빼줍니다 (Overflow 방지).
- 최종적으로 예측값과 caches를 반환합니다.

---

## **손실 함수 (크로스 엔트로피 손실)**

```python
def compute_loss(predictions, targets):
    loss = -np.mean(np.sum(targets * np.log(predictions + EPSILON), axis=1))
    return loss
```

- 예측값과 실제 타깃을 비교하여 손실을 계산하는 함수입니다.
    - 모델의 예측이 얼마나 틀렸는지 확인하고자 사용하였습니다.
- 신경망의 학습 과정에서 오차를 측정하고, 가중치 업데이트를 위한 그레디언트 계산의 기반이 됩다
- 다중 클래스 분류에서 일반적으로 사용하는 크로스 엔트로피 손실을 사용합니다.
    - 분류 문제에서 적합:
        - 이 손실 함수는 확률 분포 간의 차이를 측정하는 데 사용됩니다.
    - Softmax 출력과 자연스럽게 연계:
        - 분류 문제에서 출력층에 Softmax 활성화 함수를 사용하는 경우, Cross-Entropy Loss가 최적화에 매우 적합하고, Softmax는 각 클래스에 대한 확률 분포를 생성하므로, 이를 정답 분포와 비교하는 Cross-Entropy Loss가 잘 어울리기에 다음과 같이 설정하였습니다.
- EPSILON을 더하여 로그 함수의 0으로 나누는 오류를 방지합니다.
- 각 샘플의 손실을 평균하여 반환합니다.

모델이 출력한 확률과 실제 정답 간의 차이를 계산하여 학습을 위한 지표로 사용하고자 하였습니다

---

## 역전파 함수

```python
def backward(predictions, targets, caches):
    grads_w = []
    grads_b = []

    # 출력층 그레디언트
    delta = (predictions - targets) / targets.shape[0]
    out_prev, w, b, _ = caches[-1]
    dw = out_prev.T @ delta + L2_LAMBDA * w
    db = np.sum(delta, axis=0, keepdims=True)
    grads_w.insert(0, dw)
    grads_b.insert(0, db)
    delta = delta @ w.T

    # 은닉층 그레디언트
    for i in range(HIDDEN_LAYERS - 1, -1, -1):
        out_prev, w, b, z = caches[i]
        da = delta * leaky_relu_derivative(z)
        dw = out_prev.T @ da + L2_LAMBDA * w
        db = np.sum(da, axis=0, keepdims=True)
        delta = da @ w.T

        grads_w.insert(0, dw)
        grads_b.insert(0, db)
    return grads_w, grads_b
```

- 역전파를 수행하여 가중치와 편향의 그레디언트를 계산하는 함수입니다**.**
- grads_w, grads_b 리스트는 각 층의 가중치와 편향에 대한 그레디언트를 저장합니다.
- 출력층부터 역으로 계산합니다.
    - 출력층의 오차 delta를 계산합니다: delta = (predictions - targets) / targets.shape[0]
    - 가중치의 그레디언트 dw는 이전 층의 출력과 delta의 곱입니다.
    - 편향의 그레디언트 db는 delta의 합입니다.
    - L2 정규화를 위해 L2_LAMBDA * w를 더해줍니다.
    - delta를 업데이트하여 다음 층으로 전파합니다.
- 은닉층에서도 같은 방식으로 계산하되, 활성화 함수의 도함수를 곱해줍니다.
- 최종적으로 각 층의 그레디언트를 반환합니다.

---

## 평가 함수

```python
def evaluate(weights, biases, test_data, test_labels, verbose=True):
    correct = 0
    total_loss = 0
    tp = np.zeros(NUM_CLASSES)  # True Positives
    fp = np.zeros(NUM_CLASSES)  # False Positives
    fn = np.zeros(NUM_CLASSES)  # False Negatives

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for i in range(test_data.shape[0]):
        input_sample = test_data[i:i+1]
        target_sample = np.eye(NUM_CLASSES)[test_labels[i:i+1]]
        predictions, _ = forward(weights, biases, input_sample)
        prediction = np.argmax(predictions, axis=1)[0]
        actual = test_labels[i]
        total_loss += compute_loss(predictions, target_sample)

        confusion_matrix[actual, prediction] += 1

        if verbose:
            print(f"Sample {i + 1}: Predicted = {animal_names[prediction]}, Actual = {animal_names[actual]}")
        
        if prediction == actual:
            correct += 1
            tp[actual] += 1
        else:
            fp[prediction] += 1
            fn[actual] += 1

    accuracy = correct / test_data.shape[0]

    # 클래스별 정밀도 및 재현율 계산
    precision_per_class = np.zeros(NUM_CLASSES)
    recall_per_class = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if tp[i] + fp[i] > 0:
            precision_per_class[i] = tp[i] / (tp[i] + fp[i])
        else:
            precision_per_class[i] = 0.0
        if tp[i] + fn[i] > 0:
            recall_per_class[i] = tp[i] / (tp[i] + fn[i])
        else:
            recall_per_class[i] = 0.0

    # 평균 정밀도 및 재현율 계산 (매크로 평균)
    average_precision = np.mean(precision_per_class)
    average_recall = np.mean(recall_per_class)

    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    for i in range(NUM_CLASSES):
        print(f"Class '{animal_names[i]}': Precision = {precision_per_class[i] * 100:.2f}%, Recall = {recall_per_class[i] * 100:.2f}%")
    print(f"\nAverage Precision: {average_precision * 100:.2f}%")
    print(f"Average Recall: {average_recall * 100:.2f}%\n")

    # Confusion Matrix 출력
    print("Confusion Matrix:")
    print("Actual \\ Predicted")
    print("          ", end="")
    for name in animal_names:
        print(f"{name:^10}", end="")
    print()
    for i, row in enumerate(confusion_matrix):
        print(f"{animal_names[i]:<10}", end="")
        for val in row:
            print(f"{val:^10}", end="")
        print()

    return accuracy, total_loss / test_data.shape[0], precision_per_class, average_precision
```

- 모델의 성능을 평가하기 위한 함수입니다.
- 정확도, 정밀도, 재현율, 혼동 행렬 등을 계산합니다.
- 각 샘플에 대해 예측값과 실제 값을 비교하고, 혼동 행렬을 업데이트합니다.
- 클래스별로 정밀도와 재현율을 계산하고 출력합니다.
- 평균 정밀도와 평균 재현율을 계산합니다.
- 최종적으로 정확도, 평균 손실, 클래스별 정밀도, 평균 정밀도를 반환합니다.

---

## 가중치와 편향 저장 함수

```python
def save_weights_and_biases(weights, biases, weight_file=r"C:\weights\weights.bin", bias_file=r"C:\biases\biases.bin"):
    # 가중치 저장
    with open(weight_file, 'wb') as wf:
        for w in weights:
            w.astype(np.float32).tofile(wf)  # 이진 형식으로 저장

    # 편향 저장
    with open(bias_file, 'wb') as bf:
        for b in biases:
            b.astype(np.float32).tofile(bf)  # 이진 형식으로 저장

    print(f"Weights saved to {weight_file}")
    print(f"Biases saved to {bias_file}")
```

- 학습된 가중치와 편향을 파일로 저장하는 함수로 c코드에서 활용하기 위해 추가한 함수입니다
- 가중치와 편향을 이진 형식으로 저장하여 저장하고자하는 경로를 지정한 뒤에 이후에 로드할 수 있도록 합니다.

처음엔 txt 파일 형식으로 진행하였다가 txt형식은 저장과 읽기 속도가 느리기도 하고, 데이터를 문자열로 변환하면서 정밀도가 떨어질 수 있는 우려 때문에 bin형식으로 저장하기로 하였습니다.

---

## 메인 학습 루프

```python
def main():
```

- 전체 학습 프로세스를 관리하는 메인 함수로, 데이터 로드, 전처리, 모델 초기화, 학습 루프, 평가 를 수행합니다.

### 데이터 로드 및 전처리

```python
train_directory = r"C:\Users\최승렬\Desktop\AI\augmented_images\train_pgm"
test_directory = r"C:\Users\최승렬\Desktop\AI\augmented_images\test_pgm"

train_filenames = [f for f in os.listdir(train_directory) if f.endswith(".pgm")]
test_filenames = [f for f in os.listdir(test_directory) if f.endswith(".pgm")]

train_data, train_labels = load_images_with_labels(train_directory, train_filenames)
test_data, test_labels = load_images_with_labels(test_directory, test_filenames)
```

- 학습 및 테스트 데이터셋 디렉토리를 지정하고, 파일명을 가져옵니다. 여기서 생성된 데이터셋의 경로를 지정해서 실험을 진행하면 됩니다.
- load_images_with_labels 함수를 사용하여 데이터를 로드합니다.

### 라벨이 없는 데이터 제거

```python
# 레벨이 -1인 경우 제거
valid_indices = train_labels != -1
train_data = train_data[valid_indices]
train_labels = train_labels[valid_indices]

valid_indices = test_labels != -1
test_data = test_data[valid_indices]
test_labels = test_labels[valid_indices]
```

- 라벨을 찾지 못한 (-1) 데이터 샘플을 제거합니다.

### 데이터 및 라벨 확인

```python
# 데이터 및 라벨 확인
for idx in range(10):
    print(f"Filename: {train_filenames[idx]}, Label: {train_labels[idx]}, Class: {animal_names[train_labels[idx]]}")

unique, counts = np.unique(train_labels, return_counts=True)
print("Training data class distribution:", dict(zip(unique, counts)))

unique, counts = np.unique(test_labels, return_counts=True)
print("Test data class distribution:", dict(zip(unique, counts)))
```

- 일부 데이터 샘플의 파일명과 라벨을 출력하여 확인합니다.
- 클래스별 데이터 분포를 출력합니다.

### 데이터 정규화

```python
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0) + EPSILON
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

np.save('C:\weights\mean.npy', mean)
np.save('C:\weights\std.npy', std)
    
mean.astype(np.float64).tofile('C:\weights\mean.bin')
std.astype(np.float64).tofile('C:\weights\std.bin') 
```

- 데이터의 평균과 표준편차를 계산하여 표준화합니다.
- 학습 데이터의 평균과 표준편차를 사용하여 테스트 데이터도 동일하게 정규화합니다.
- np.save 사용하여mean과 std를 Numpy의 바이너리 형식으로 저장하였습니다.
- tofile 사용하여 mean과 std를 .bin 파일로 저장. 이진 데이터로 저장하며, 저장 전에 float64 타입으로 변환하였고, 이를 C코드에서 정규화 데이터를 사용할 때 활용합니다. 각각 저장하고자 하는 경로를 지정해주면 됩니다.

### 모델 초기화

```python
weights, biases = init_network()
```

- 신경망의 가중치와 편향을 초기화합니다.

### 손실 추적을 위한 리스트 초기화

```python
train_losses = []
val_losses = []
```

- 에폭별로 손실 값을 저장하기 위한 리스트를 초기화합니다.

### Adam 옵티마이저 파라미터 초기화

```python
m_w = [np.zeros_like(w) for w in weights]
v_w = [np.zeros_like(w) for w in weights]
m_b = [np.zeros_like(b) for b in biases]
v_b = [np.zeros_like(b) for b in biases]
beta1 = 0.9
beta2 = 0.999
epsilon = EPSILON
```

- 최적화 방법은 adam을 사용하였는데 이미지 분류와 같은 작업에서도 많이 사용되기도 하고, 학습이 불안정할 수 있는 상황에서 안정성을 보장한다고 알려져 경사하강법에서 바꾸어 사용해보고자 하였습니다.
- Adam 옵티마이저에 필요한 1차 및 2차 모멘트 벡터를 초기화합니다.
- beta1, beta2는 모멘트 추적을 위한 하이퍼파라미터로 저 값들을 조정하면서 실험을 진행하였습니다.

### 학습 루프

```python
for epoch in range(EPOCHS):
    learning_rate = INITIAL_LEARNING_RATE
    print(f"\nEpoch {epoch + 1}/{EPOCHS} - Learning Rate: {learning_rate:.6f}")
    epoch_loss = 0

    # 셔플 데이터
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    for step in range(0, len(train_data), BATCH_SIZE):
        # 코드 내용 생략
```

- 에폭 수만큼 반복하여 학습합니다.
- 각 에폭마다 학습률을 설정하고, 에폭 손실을 초기화합니다.
- 데이터를 셔플하여 미니배치가 다양하게 구성되도록 합니다.
- 미니배치 단위로 데이터를 처리합니다.

### 순전파, 손실 계산, 역전파

```python
# 순전파
predictions, caches = forward(weights, biases, input_batch)

# 손실 계산
batch_loss = compute_loss(predictions, target_batch)
epoch_loss += batch_loss

# 역전파
grads_w, grads_b = backward(predictions, target_batch, caches)
```

- 입력 배치를 모델에 통과시켜 예측값을 얻습니다.
- 예측값과 실제 타깃을 비교하여 손실을 계산하고, 에폭 손실에 누적합니다.
- 역전파를 수행하여 그레디언트를 계산합니다.

### 그레디언트 클리핑

```python
for i in range(len(grads_w)):
    grad_norm_w = np.linalg.norm(grads_w[i])
    if grad_norm_w > MAX_GRAD_NORM:
        grads_w[i] = grads_w[i] * (MAX_GRAD_NORM / grad_norm_w)
    grad_norm_b = np.linalg.norm(grads_b[i])
    if grad_norm_b > MAX_GRAD_NORM:
        grads_b[i] = grads_b[i] * (MAX_GRAD_NORM / grad_norm_b)
```

- 그레디언트의 노름이 MAX_GRAD_NORM을 넘을 경우 스케일링하여 클리핑합니다.

실험을 진행할 때 학습률을 높게 하거나, 은닉 층의 레이어 수와 노드 수를 높게 해서 진행할 때, 또는 데이터셋2를 진행할 때 그레디언트 노름이 폭증하는 경우를 확인할 수 있었습니다.

그레디언트 폭증은 손실 값이 발산하거나, 매우 불안정한 학습 패턴이 나타나기도 하고, 높은 손실 값으로 인해 모델의 성능이 저하되거나 학습 과정에서 가중치 값이 매우 커지거나 비정상적으로 변하기에 특정 값을 초과하는 그레디언트를 임계값으로 정규화해 학습 안정성을 보장하고자 하였습니다

- 이는 그레디언트 폭주를 방지하고 안정적인 학습을 도와줍니다.

### 그레디언트 노름 출력

```python
grad_norm = np.linalg.norm(grads_w[-1])
print(f"Epoch {epoch + 1}, Batch {step // BATCH_SIZE + 1}: Grad Norm = {grad_norm:.6f}")
```

- 마지막 층의 가중치 그레디언트 노름을 계산하여 출력합니다.
- 학습 과정 중 그레디언트의 변화를 모니터링할 수 있습니다.

### Adam 옵티마이저를 사용한 파라미터 업데이트

```python
for i in range(len(weights)):
    # 모멘트 업데이트
    m_w[i] = beta1 * m_w[i] + (1 - beta1) * grads_w[i]
    v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grads_w[i] ** 2)
    m_b[i] = beta1 * m_b[i] + (1 - beta1) * grads_b[i]
    v_b[i] = beta2 * v_b[i] + (1 - beta2) * (grads_b[i] ** 2)

    # 모멘트 편향 보정
    m_w_hat = m_w[i] / (1 - beta1 ** (epoch + 1))
    v_w_hat = v_w[i] / (1 - beta2 ** (epoch + 1))
    m_b_hat = m_b[i] / (1 - beta1 ** (epoch + 1))
    v_b_hat = v_b[i] / (1 - beta2 ** (epoch + 1))

    # 가중치 및 편향 업데이트
    weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
    biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
```

- Adam 옵티마이저의 업데이트 규칙을 적용하여 가중치와 편향을 업데이트합니다.
- 1차 모멘트(m_w, m_b)와 2차 모멘트(v_w, v_b)를 업데이트합니다.
- 모멘트 편향 보정을 수행하여 초기 단계의 편향을 제거합니다.
- 학습률과 함께 가중치와 편향을 업데이트합니다.

### 에폭 손실 저장 및 평가

```python
train_losses.append(epoch_loss / (len(train_data) // BATCH_SIZE))
_, val_loss, _, _ = evaluate(weights, biases, test_data, test_labels, verbose=False)
val_losses.append(val_loss)

print(f"Epoch {epoch + 1} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
```

- 에폭이 끝날 때마다 평균 손실을 계산하여 저장합니다.
- evaluate 함수를 사용하여 검증 데이터에 대한 손실을 계산합니다.
- 학습 손실과 검증 손실을 출력하여 학습 과정을 모니터링하고자 했습니다.

### 샘플 출력 확률 분포 확인

```python
# 일부 샘플에 대한 출력 확률 분포 확인
sample_output, _ = forward(weights, biases, test_data[0:5])
print(f"Sample Output Probabilities (First 5 Samples):\n{sample_output}"
```

- 테스트 데이터의 첫 5개 샘플에 대한 예측 확률 분포를 출력합니다.
- 모델이 어떻게 예측하고 있는지 확인하고자 했습니다.

### 가중치와 편향 저장

```python
save_weights_and_biases(weights, biases)
```

- 학습이 완료된 후 가중치와 편향을 저장해서 후에 c코드에서 평가할 때 사용합니다.

### 학습 및 검증 손실 그래프 시각화

```python
# 학습 및 검증 손실 그래프 시각화
plt.plot(range(EPOCHS), train_losses, label='Training Loss')
plt.plot(range(EPOCHS), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

print("\nFinal evaluation:")
evaluate(weights, biases, test_data, test_labels, verbose=True)
```

- 에폭별 학습 손실과 검증 손실을 그래프로 시각화합니다.
- 학습 과정 중 손실의 변화를 시각적으로 확인할 수 있습니다.
- 전체 테스트 데이터에 대해 최종 평가를 수행하고, 자세한 결과를 출력합니다.
