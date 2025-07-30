import numpy as np
import os
import matplotlib.pyplot as plt

# 상수 정의
NUM_CLASSES = 4
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 0.0001
EPOCHS = 60
INPUT_NODES = 256 * 256
HIDDEN_LAYERS = 6
HIDDEN_NODES = [128,128,128,64,64,64]
OUTPUT_NODES = NUM_CLASSES
L2_LAMBDA = 0.001
EPSILON = 1e-5

# 그레이디언트 클리핑 임계값
MAX_GRAD_NORM = 8.0

# 클래스 라벨
animal_names = ["cat", "dog", "tiger", "hyena"]

# 데이터 로드 함수 정의
def load_pgm_image(file_path):
    with open(file_path, 'rb') as file:
        file.readline()  # "P5"
        file.readline()  # width height
        file.readline()  # max_val
        data = np.fromfile(file, dtype=np.uint8)
        data = data.reshape((256, 256)).astype(np.float32) / 255.0
    return data.flatten()

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
    return -1  # 레이블을 찾지 못한 경우

# 활성화 함수 정의
def leaky_relu(x, alpha=0.0001):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.0001):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# He 초기화 함수
def he_initialize(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2.0 / rows)

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

# 손실 함수 (Cross Entropy Loss)
def compute_loss(predictions, targets):
    loss = -np.mean(np.sum(targets * np.log(predictions + EPSILON), axis=1))
    return loss

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

# 메인 학습 루프
def main():
    train_directory = r"C:\Users\최승렬\Desktop\AI\augmented_images\train_pgm"
    test_directory = r"C:\Users\최승렬\Desktop\AI\augmented_images\test_pgm"

    train_filenames = [f for f in os.listdir(train_directory) if f.endswith(".pgm")]
    test_filenames = [f for f in os.listdir(test_directory) if f.endswith(".pgm")]

    train_data, train_labels = load_images_with_labels(train_directory, train_filenames)
    test_data, test_labels = load_images_with_labels(test_directory, test_filenames)

    # 레이블이 -1인 경우(레이블을 찾지 못한 경우) 제거
    valid_indices = train_labels != -1
    train_data = train_data[valid_indices]
    train_labels = train_labels[valid_indices]

    valid_indices = test_labels != -1
    test_data = test_data[valid_indices]
    test_labels = test_labels[valid_indices]

    # 데이터 및 레이블 확인
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
