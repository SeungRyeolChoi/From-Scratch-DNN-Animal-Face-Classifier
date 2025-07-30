본 실험에서는 고양이, 개, 호랑이, 하이에나 네 종류의 동물 얼굴을 대상으로 심층망을 설계하고, 이를 통해 최적으로 동물 얼굴을 인식하는 모델을 구현하고자 한다. 본 실험의 주요 목표는 데이터 증강 기법을 통해 학습 데이터를 강화하고, 최적의 신경망 구조를 c와 python으로 구성하고, 여러 하이퍼파라미터 설정을 찾아가는 비교 실험을 통해 동물 얼굴 인식 정확도를 극대화하는 것이다. 

이를 위해 이미지의 크기 조정, 회전, 반전, 노이즈 추가 등 다양한 데이터 증강 기법을 적용하여 데이터셋을 구성하고, 모델의 일반화 능력을 향상시키고자 하였다. 데이터 증강을 통한 데이터의 다양성 확보, 최적의 신경망 구조 설계 및 하이퍼파라미터 튜닝을 통해 동물의 얼굴을 효과적으로 인식할 수 있는 신경망을 구축하고자 한다


```bash
import numpy as np
import os
import matplotlib.pyplot as plt
```
배열 및 행렬 연산을 사용하고자 수치 연산을 위한 numpy라이브러리를 사용하였습니다.
운영체제와 상호작용하기 위한 모듈로, 파일 및 디렉토리 경로 조작에 사용되는 os라이브러리를 사용했습니다
그래프와 시각화를 위해 matplotlib.pyplot 라이브러리를 사용했습니다

```bash
# 상수 정의
NUM_CLASSES = 4
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 0.001
EPOCHS = 60
INPUT_NODES = 256 * 256
HIDDEN_LAYERS = 6
HIDDEN_NODES = [128,128,128,64,64,64]
OUTPUT_NODES = NUM_CLASSES
L2_LAMBDA = 0.001
EPSILON = 1e-5 
```

```bash
# 그레이디언트 클리핑 임계값
MAX_GRAD_NORM = 8.0
```

# 클래스 라벨
animal_names = ["cat", "dog", "tiger", "hyena"]
NUM_CLASSES: 분류할 객체의 수로, 4개의 동물 클래스입니다.
BATCH_SIZE: 한 번의 학습 단계에서 사용할 데이터 샘플의 수로 32로 설정했습니다.
INITIAL_LEARNING_RATE: 초기 학습률입니다.
EPOCHS: 전체 데이터셋을 몇 번 반복하여 학습할 것인지 결정하는 에폭입니다.
INPUT_NODES: 입력층의 노드 수로, 신경망의 입력층에 전달되는 데이터의 크기, 이미지의 픽셀 수입니다.
HIDDEN_LAYERS: 은닉층의 수입니다.
HIDDEN_NODES: 각 은닉층의 노드 수를 리스트로 정의하였습니다.
OUTPUT_NODES: 출력층의 노드 수로 4개의 클래스 중 하나로 분류해야 하기에 다음과 같이 설정하였다.
L2_LAMBDA: L2 정규화의 람다 값으로, 과적합을 방지하기 위해 사용하였습니다.
EPSILON: 작은 값으로, 계산 중에 0으로 나누는 오류를 방지하고자 사용하였습니다.
MAX_GRAD_NORM: 그레이디언트 클리핑을 위한 임계값으로, 그레이디언트 폭주를 방지하고자 사용하였습니다.
animal_names: 클래스 라벨을 정의한 리스트입니다.
# 데이터 로드 함수 정의
def load_pgm_image(file_path):
    with open(file_path, 'rb') as file:
        file.readline()  # "P5"
        file.readline()  # width height
        file.readline()  # max_val
        data = np.fromfile(file, dtype=np.uint8)
        data = data.reshape((256, 256)).astype(np.float32) / 255.0
    return data.flatten()
데이터셋의 이미지들이 pgm형식으로 되어있는데, pgm 이미지 파일을 읽어와서 일차원 배열로 반환하는 함수입니다.
바이너리 모드로 파일을 열고, 헤더 부분("P5", 이미지 크기, 최대 값)을 읽어 넘깁니다.
픽셀 데이터를 np.uint8 타입으로 읽어옵니다.
이미지를 (256, 256) 형태로 재구성하고, float32로 변환 후 0~1 사이로 정규화합니다.
이미지를 평탄화하여 일차원 배열로 반환합니다.
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

def get_label_from_filename(filename):
    filename = filename.lower() 
    for i, name in enumerate(animal_names):
        if name in filename:
            return i
    print(f"Warning: Label not found for file {filename}")
    return -1  # 라벨을 찾지 못한 경우
데이터셋의 이미지들이 pgm형식으로 되어있는데, pgm 이미지 파일을 읽어와서 일차원 배열로 반환하는 함수입니다.

바이너리 모드로 파일을 열고, 헤더 부분("P5", 이미지 크기, 최대 값)을 읽어 넘깁니다.

픽셀 데이터를 np.uint8 타입으로 읽어옵니다.

이미지를 (256, 256) 형태로 재구성하고, float32로 변환 후 0~1 사이로 정규화합니다.

이미지를 평탄화하여 일차원 배열로 반환합니다.

파일명에서 클래스 라벨을 추출하는 함수입니다.

데이터셋 증강을 위해 이미지들을 조정하다보니 이미지의 이름들에서 class 말고 cat_24_rot-15 이런식으로 되어있기에 클래스 라벨을 추출할 필요성을 느껴서 추가하였습니다.

이미지들 이름에서 클래스들이 전부 소문자로 되어있기에 파일명을 소문자로 변환하여 일관성을 유지하고자 하였습니다.

animal_names 리스트를 순회하며 파일명에 해당 동물명이 포함되어 있는지 확인합니다.

해당 동물명이 포함되어 있으면 그 인덱스를 라벨로 반환합니다.

해당되지 않는 경우 -1을 반환하고 경고 메시지를 출력합니다.

# 활성화 함수 정의
def leaky_relu(x, alpha=0.0001):
    return np.where(x > 0, x, alpha * x)
Leaky ReLU 활성화 함수를 구현하였습니다.
입력 x가 양수이면 그대로 반환하고, 음수이면 alpha를 곱하여 기울기를 유지합니다.
alpha는 음수 영역의 기울기로 기본값은 0.01로 설정하였습니다.
def leaky_relu_derivative(x, alpha=0.0001):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx
Leaky ReLU의 도함수를 계산하는 함수로 역전파 시 그레이디언트 계산에 사용됩니다.
입력 x가 양수이면 도함수는 1, 음수이면 alpha입니다.
# He 초기화 함수
def he_initialize(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2.0 / rows)
가중치를 초기화 하는 함수로 He 초기화 함수를 선택하였습니다. 깊은 신경망에서의 기울기 소실 문제를 완화할 수 있는 역할을 합니다.
표준 정규분포에서 난수를 생성하고, 분산을 2.0 / rows로 조정합니다.
rows는 입력 노드 수, cols는 출력 노드 수입니다.
# 네트워크 초기화 함수
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
신경망의 가중치와 편향을 초기화하는 함수입니다.
은닉층의 수만큼 루프를 돌며 각 층의 가중치와 편향을 초기화합니다.
첫 번째 층의 입력 노드 수는 INPUT_NODES이며, 그 이후 층은 이전 층의 노드 수를 사용합니다.
가중치는 he_initialize 함수를 사용하여 초기화하고, 편향은 0으로 초기화합니다.
출력층의 가중치와 편향도 동일하게 초기화합니다
# 순전파 함수
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
순전파를 수행하여 입력 데이터로부터 예측값을 계산하는 함수입니다.
caches 리스트는 역전파 시 필요한 중간 계산 값을 저장합니다.{효율적인 학습(역전파)을 위해}
각 은닉층에 대해:
z는 선형 변환 결과이며, z = out @ weights[i] + biases[i]로 계산됩니다.
a는 활성화 함수를 적용한 결과입니다.
caches에 입력, 가중치, 편향, z를 저장합니다.
out은 다음 층의 입력이 됩니다.
출력층에서:
선형 변환 후 softmax 함수를 적용하여 클래스 확률을 계산합니다.
안정성을 위해 z에서 최대값을 빼줍니다 (Overflow 방지).
최종적으로 예측값과 caches를 반환합니다.
# 손실 함수 (Cross Entropy Loss)
def compute_loss(predictions, targets):
    loss = -np.mean(np.sum(targets * np.log(predictions + EPSILON), axis=1))
    return loss
예측값과 실제 타깃을 비교하여 손실을 계산하는 함수입니다.
모델의 예측이 얼마나 틀렸는지 확인하고자 사용하였습니다.
신경망의 학습 과정에서 오차를 측정하고, 가중치 업데이트를 위한 그레이디언트 계산의 기반이 됩다
다중 클래스 분류에서 일반적으로 사용하는 크로스 엔트로피 손실을 사용합니다.
분류 문제에서 적합:
이 손실 함수는 확률 분포 간의 차이를 측정하는 데 사용됩니다.
Softmax 출력과 자연스럽게 연계:
분류 문제에서 출력층에 Softmax 활성화 함수를 사용하는 경우, Cross-Entropy Loss가 최적화에 매우 적합하고, Softmax는 각 클래스에 대한 확률 분포를 생성하므로, 이를 정답 분포와 비교하는 Cross-Entropy Loss가 잘 어울리기에 다음과 같이 설정하였습니다.
EPSILON을 더하여 로그 함수의 0으로 나누는 오류를 방지합니다.
각 샘플의 손실을 평균하여 반환합니다.
# 역전파 함수
def backward(predictions, targets, caches):
    grads_w = []
    grads_b = []

    # 출력층 그레이디언트
    delta = (predictions - targets) / targets.shape[0]
    out_prev, w, b, _ = caches[-1]
    dw = out_prev.T @ delta + L2_LAMBDA * w
    db = np.sum(delta, axis=0, keepdims=True)
    grads_w.insert(0, dw)
    grads_b.insert(0, db)
    delta = delta @ w.T

    # 은닉층 그레이디언트
    for i in range(HIDDEN_LAYERS - 1, -1, -1):
        out_prev, w, b, z = caches[i]
        da = delta * leaky_relu_derivative(z)
        dw = out_prev.T @ da + L2_LAMBDA * w
        db = np.sum(da, axis=0, keepdims=True)
        delta = da @ w.T

        grads_w.insert(0, dw)
        grads_b.insert(0, db)
    return grads_w, grads_b
역전파를 수행하여 가중치와 편향의 그레이디언트를 계산하는 함수입니다.
grads_w, grads_b 리스트는 각 층의 가중치와 편향에 대한 그레이디언트를 저장합니다.
출력층부터 역으로 계산합니다.
출력층의 오차 delta를 계산합니다: delta = (predictions - targets) / targets.shape[0]
가중치의 그레이디언트 dw는 이전 층의 출력과 delta의 곱입니다.
편향의 그레이디언트 db는 delta의 합입니다.
L2 정규화를 위해 L2_LAMBDA * w를 더해줍니다.
delta를 업데이트하여 다음 층으로 전파합니다.
은닉층에서도 같은 방식으로 계산하되, 활성화 함수의 도함수를 곱해줍니다.
최종적으로 각 층의 그레이디언트를 반환합니다.
# 평가 함수
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

    # 평균 정밀도 및 재현율 계산
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
모델의 성능을 평가하기 위한 함수입니다.
정확도, 정밀도, 재현율, 혼동 행렬 등을 계산합니다.
각 샘플에 대해 예측값과 실제 값을 비교하고, 혼동 행렬을 업데이트합니다.
클래스별로 정밀도와 재현율을 계산하고 출력합니다.
평균 정밀도와 평균 재현율을 계산합니다.
최종적으로 정확도, 평균 손실, 클래스별 정밀도, 평균 정밀도를 반환합니다.
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
학습된 가중치와 편향을 파일로 저장하는 함수로 c코드에서 활용하기 위해 추가한 함수입니다.
가중치와 편향을 이진 형식으로 저장하여 저장하고자하는 경로를 지정한 뒤에 이후에 로드할 수 있도록 합니다.
# 메인 학습 루프
def main():
    train_directory = r"C:\Users\최승렬\Desktop\AI\augmented_images\train_pgm"
    test_directory = r"C:\Users\최승렬\Desktop\AI\augmented_images\test_pgm"

    train_filenames = [f for f in os.listdir(train_directory) if f.endswith(".pgm")]
    test_filenames = [f for f in os.listdir(test_directory) if f.endswith(".pgm")]

    train_data, train_labels = load_images_with_labels(train_directory, train_filenames)
    test_data, test_labels = load_images_with_labels(test_directory, test_filenames)

    # 라벨이 -1인 경우(라벨을 찾지 못한 경우) 제거
    valid_indices = train_labels != -1
    train_data = train_data[valid_indices]
    train_labels = train_labels[valid_indices]

    valid_indices = test_labels != -1
    test_data = test_data[valid_indices]
    test_labels = test_labels[valid_indices]

    # 데이터 및 라벨 확인
    for idx in range(10):
        print(f"Filename: {train_filenames[idx]}, Label: {train_labels[idx]}, Class: {animal_names[train_labels[idx]]}")

    unique, counts = np.unique(train_labels, return_counts=True)
    print("Training data class distribution:", dict(zip(unique, counts)))

    unique, counts = np.unique(test_labels, return_counts=True)
    print("Test data class distribution:", dict(zip(unique, counts)))

    # 데이터 정규화
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + EPSILON
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    np.save('C:\weights\mean.npy', mean)
    np.save('C:\weights\std.npy', std)
    
    mean.astype(np.float64).tofile('C:\weights\mean.bin')
    std.astype(np.float64).tofile('C:\weights\std.bin') 
    
    weights, biases = init_network()

    train_losses = []
    val_losses = []

    # Adam 옵티마이저 파라미터 초기화
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    beta1 = 0.5
    beta2 = 0.9
    epsilon = EPSILON

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
            input_batch = train_data[step:step + BATCH_SIZE]
            target_batch_indices = train_labels[step:step + BATCH_SIZE]
            target_batch = np.eye(NUM_CLASSES)[target_batch_indices]

            # 순전파
            predictions, caches = forward(weights, biases, input_batch)

            # 손실 계산
            batch_loss = compute_loss(predictions, target_batch)
            epoch_loss += batch_loss

            # 역전파
            grads_w, grads_b = backward(predictions, target_batch, caches)

            # 그레이디언트 클리핑 적용
            for i in range(len(grads_w)):
                grad_norm_w = np.linalg.norm(grads_w[i])
                if grad_norm_w > MAX_GRAD_NORM:
                    grads_w[i] = grads_w[i] * (MAX_GRAD_NORM / grad_norm_w)
                grad_norm_b = np.linalg.norm(grads_b[i])
                if grad_norm_b > MAX_GRAD_NORM:
                    grads_b[i] = grads_b[i] * (MAX_GRAD_NORM / grad_norm_b)

            # 그레이디언트 노름 출력
            grad_norm = np.linalg.norm(grads_w[-1])
            print(f"Epoch {epoch + 1}, Batch {step // BATCH_SIZE + 1}: Grad Norm = {grad_norm:.6f}")

            # Adam 옵티마이저를 사용한 가중치 및 편향 업데이트
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

        train_losses.append(epoch_loss / (len(train_data) // BATCH_SIZE))
        _, val_loss, _, _ = evaluate(weights, biases, test_data, test_labels, verbose=False)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

        # 일부 샘플에 대한 출력 확률 분포 확인
        sample_output, _ = forward(weights, biases, test_data[0:5])
        print(f"Sample Output Probabilities (First 5 Samples):\n{sample_output}")

    save_weights_and_biases(weights, biases)
        
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

if __name__ == "__main__":
    main()
전체 학습 프로세스를 관리하는 메인 함수로, 데이터 로드, 전처리, 모델 초기화, 학습 루프, 평가 를 수행합니다.

학습 및 테스트 데이터셋 디렉토리를 지정하고, 파일명을 가져옵니다. 여기서 생성된 데이터셋의 경로를 지정해서 실험을 진행하면 됩니다.
load_images_with_labels 함수를 사용하여 데이터를 로드합니다.
라벨을 찾지 못한 (-1) 데이터 샘플을 제거합니다.
일부 데이터 샘플의 파일명과 라벨을 출력하여 확인합니다.
클래스별 데이터 분포를 출력합니다.
데이터의 평균과 표준편차를 계산하여 표준화합니다.
학습 데이터의 평균과 표준편차를 사용하여 테스트 데이터도 동일하게 정규화합니다.
np.save 사용하여mean과 std를 Numpy의 바이너리 형식으로 저장하였습니다.
tofile 사용하여 mean과 std를 .bin 파일로 저장. 이진 데이터로 저장하며, 저장 전에 float64 타입으로 변환하였고, 이를 C코드에서 정규화 데이터를 사용할 때 활용합니다. 각각 저장하고자 하는 경로를 지정해주면 됩니다.
신경망의 가중치와 편향을 초기화합니다.
에폭별로 손실 값을 저장하기 위한 리스트를 초기화합니다.
에폭 수만큼 반복하여 학습합니다.
각 에폭마다 학습률을 설정하고, 에폭 손실을 초기화합니다.
데이터를 셔플하여 미니배치가 다양하게 구성되도록 합니다.
미니배치 단위로 데이터를 처리합니다.
입력 배치를 모델에 통과시켜 예측값을 얻습니다.
예측값과 실제 타깃을 비교하여 손실을 계산하고, 에폭 손실에 누적합니다.
역전파를 수행하여 그레이디언트를 계산합니다.
그레이디언트의 노름이 MAX_GRAD_NORM을 넘을 경우 스케일링하여 클리핑합니다.
마지막 층의 가중치 그레이디언트 노름을 계산하여 출력합니다.
학습 과정 중 그레이디언트의 변화를 모니터링할 수 있습니다.
Adam 옵티마이저의 업데이트 규칙을 적용하여 가중치와 편향을 업데이트합니다.
1차 모멘트(m_w, m_b)와 2차 모멘트(v_w, v_b)를 업데이트합니다.
모멘트 편향 보정을 수행하여 초기 단계의 편향을 제거합니다.
학습률과 함께 가중치와 편향을 업데이트합니다.
에폭이 끝날 때마다 평균 손실을 계산하여 저장합니다.
evaluate 함수를 사용하여 검증 데이터에 대한 손실을 계산합니다.
학습 손실과 검증 손실을 출력하여 학습 과정을 모니터링하고자 했습니다.
테스트 데이터의 첫 5개 샘플에 대한 예측 확률 분포를 출력합니다.
모델이 어떻게 예측하고 있는지 확인하고자 했습니다.
학습이 완료된 후 가중치와 편향을 저장해서 후에 c코드에서 평가할 때 사용합니다.
에폭별 학습 손실과 검증 손실을 그래프로 시각화합니다.
학습 과정 중 손실의 변화를 시각적으로 확인할 수 있습니다.
전체 테스트 데이터에 대해 최종 평가를 수행하고, 자세한 결과를 출력합니다.
