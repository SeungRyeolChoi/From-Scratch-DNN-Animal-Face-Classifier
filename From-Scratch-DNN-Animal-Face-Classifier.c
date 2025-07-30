#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_CLASSES 4
#define INPUT_NODES 65536  // 256x256 pixels
#define HIDDEN_LAYERS 6
#define OUTPUT_NODES NUM_CLASSES
#define MAX_FILENAME 256
#define NUM_SAMPLES 168
#define DIRECTORY L"C:\\test_pgm\\*.pgm"

int hidden_nodes[HIDDEN_LAYERS] = {128,128,128,64,64,64};
wchar_t* class_names[NUM_CLASSES] = { L"cat", L"dog", L"tiger", L"hyena" };

double leaky_relu(double x) {
    double alpha = 0.0001;
    return x > 0 ? x : alpha * x;
}

void softmax(double* output, int n) {
    double max = output[0];
    for (int i = 1; i < n; i++) {
        if (output[i] > max) {
            max = output[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        output[i] = exp(output[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

int read_pgm(const wchar_t* filename, double* data) {
    FILE* file = _wfopen(filename, L"rb");
    if (!file) {
        fwprintf(stderr, L"Cannot open file %s\n", filename);
        return -1;
    }

    char header[3] = { 0 };
    int width, height, maxval;
    if (fscanf(file, "%2s %d %d %d", header, &width, &height, &maxval) != 4 || strcmp(header, "P5") != 0) {
        fclose(file);
        fwprintf(stderr, L"Invalid PGM file format: %s\n", filename);
        return -1;
    }

    if (width * height != INPUT_NODES) {
        fclose(file);
        fwprintf(stderr, L"Invalid image dimensions: %s\n", filename);
        return -1;
    }

    unsigned char* buffer = malloc(INPUT_NODES);
    if (!buffer) {
        fclose(file);
        fwprintf(stderr, L"Memory allocation failed for buffer.\n");
        return -1;
    }

    if (fread(buffer, 1, INPUT_NODES, file) != INPUT_NODES) {
        free(buffer);
        fclose(file);
        fwprintf(stderr, L"Failed to read pixel data: %s\n", filename);
        return -1;
    }

    for (int i = 0; i < INPUT_NODES; i++) {
        data[i] = buffer[i] / 255.0;
    }

    free(buffer);
    fclose(file);
    return 0;
}

int get_class_label(const wchar_t* filename) {
    wchar_t extracted_class[MAX_FILENAME];
    const wchar_t* pos = wcsstr(filename, L"_");
    if (!pos) {
        wprintf(L"Invalid filename format: %s\n", filename);
        return -1;
    }

    int length = pos - filename;
    wcsncpy(extracted_class, filename, length);
    extracted_class[length] = L'\0';

    for (int i = 0; i < NUM_CLASSES; i++) {
        if (_wcsicmp(extracted_class, class_names[i]) == 0) {
            return i;
        }
    }

    wprintf(L"Label not found for file: %s\n", filename);
    return -1;
}

void load_weights_and_biases(double** weights, double** biases, const char* weights_path, const char* biases_path) {
    FILE* file = fopen(weights_path, "rb");
    if (!file) {
        perror("Failed to open weights file");
        exit(EXIT_FAILURE);
    }

    int previous_nodes = INPUT_NODES;

    // weights 읽기
    for (int i = 0; i < HIDDEN_LAYERS + 1; i++) {
        int current_nodes = (i == HIDDEN_LAYERS) ? OUTPUT_NODES : hidden_nodes[i];
        float* temp_buffer = malloc(previous_nodes * current_nodes * sizeof(float)); // float 크기로 읽기
        if (!temp_buffer) {
            perror("Memory allocation failed for temporary buffer");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        size_t read_count = fread(temp_buffer, sizeof(float), previous_nodes * current_nodes, file);
        if (read_count != previous_nodes * current_nodes) {
            fprintf(stderr, "Error reading weights for layer %d: Expected %d, got %zu\n",
                i, previous_nodes * current_nodes, read_count);
            free(temp_buffer);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // float 데이터를 double로 변환
        weights[i] = malloc(previous_nodes * current_nodes * sizeof(double));
        for (int j = 0; j < previous_nodes * current_nodes; j++) {
            weights[i][j] = (double)temp_buffer[j];
        }

        free(temp_buffer);
        previous_nodes = current_nodes;

        // 디버깅: weights 값 일부 출력
        for (int j = 0; j < 5; j++) {  // 각 레이어에서 처음 5개의 weight 출력
            printf("Weights[%d][%d]: %f\n", i, j, weights[i][j]);
        }
    }
    fclose(file);

    // biases 읽기
    file = fopen(biases_path, "rb");
    if (!file) {
        perror("Failed to open biases file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < HIDDEN_LAYERS + 1; i++) {
        int current_nodes = (i == HIDDEN_LAYERS) ? OUTPUT_NODES : hidden_nodes[i];
        float* temp_buffer = malloc(current_nodes * sizeof(float)); // float 크기로 읽기
        if (!temp_buffer) {
            perror("Memory allocation failed for temporary buffer");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        size_t read_count = fread(temp_buffer, sizeof(float), current_nodes, file);
        if (read_count != current_nodes) {
            fprintf(stderr, "Error reading biases for layer %d: Expected %d, got %zu\n",
                i, current_nodes, read_count);
            free(temp_buffer);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // float 데이터를 double로 변환
        biases[i] = malloc(current_nodes * sizeof(double));
        for (int j = 0; j < current_nodes; j++) {
            biases[i][j] = (double)temp_buffer[j];
        }

        free(temp_buffer);
    }
    fclose(file);
}


void load_mean_std(double* mean, double* std, const char* mean_path, const char* std_path) {
    FILE* mean_file = fopen(mean_path, "rb");
    FILE* std_file = fopen(std_path, "rb");
    if (!mean_file || !std_file) {
        perror("Failed to open mean or std file");
        exit(EXIT_FAILURE);
    }

    if (fread(mean, sizeof(double), INPUT_NODES, mean_file) != INPUT_NODES) {
        fprintf(stderr, "Error reading mean file.\n");
        exit(EXIT_FAILURE);
    }
    if (fread(std, sizeof(double), INPUT_NODES, std_file) != INPUT_NODES) {
        fprintf(stderr, "Error reading std file.\n");
        exit(EXIT_FAILURE);
    }

    fclose(mean_file);
    fclose(std_file);
}



void forward(double* input, double** weights, double** biases, double* output) {
    double* current_output = malloc(INPUT_NODES * sizeof(double));
    if (!current_output) {
        perror("Memory allocation failed for forward pass");
        exit(EXIT_FAILURE);
    }
    memcpy(current_output, input, INPUT_NODES * sizeof(double));

    int previous_nodes = INPUT_NODES;

    for (int i = 0; i < HIDDEN_LAYERS + 1; i++) {
        int current_nodes = (i == HIDDEN_LAYERS) ? OUTPUT_NODES : hidden_nodes[i];
        double* next_output = calloc(current_nodes, sizeof(double));
        if (!next_output) {
            perror("Memory allocation failed for next output");
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < current_nodes; j++) {
            for (int k = 0; k < previous_nodes; k++) {
                next_output[j] += current_output[k] * weights[i][k * current_nodes + j];
            }
            next_output[j] += biases[i][j];
            if (i < HIDDEN_LAYERS) {
                next_output[j] = leaky_relu(next_output[j]);
            }
        }
        free(current_output);
        current_output = next_output;
        previous_nodes = current_nodes;
    }

    softmax(current_output, OUTPUT_NODES);
    memcpy(output, current_output, OUTPUT_NODES * sizeof(double));
    free(current_output);
}


void load_test_data(const wchar_t* directory, double** inputs, int* labels, int* loaded_samples) {
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile(directory, &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        fwprintf(stderr, L"FindFirstFile failed (%d)\n", GetLastError());
        *loaded_samples = 0;
        return;
    }

    int index = 0;
    do {
        if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            wchar_t filepath[MAX_FILENAME];
            swprintf(filepath, MAX_FILENAME, L"C:\\test_pgm\\%s", findFileData.cFileName);
            wprintf(L"Processing file: %s\n", filepath);  // 파일 처리 로그 추가
            labels[index] = get_class_label(findFileData.cFileName);

            if (labels[index] == -1) {
                wprintf(L"Warning: Label extraction failed for %s\n", findFileData.cFileName);
            }
            else {
                wprintf(L"File %s labeled as %s\n", findFileData.cFileName, class_names[labels[index]]);
            }

            inputs[index] = malloc(INPUT_NODES * sizeof(double));
            if (read_pgm(filepath, inputs[index]) == 0) {
                index++;
            }
            else {
                free(inputs[index]);
                wprintf(L"Failed to read: %s\n", filepath);  // 파일 읽기 실패 로그
            }
        }
    } while (FindNextFile(hFind, &findFileData) != 0 && index < NUM_SAMPLES);

    FindClose(hFind);
    *loaded_samples = index;
    wprintf(L"Total loaded samples: %d\n", *loaded_samples);  // 총 로드된 샘플 수 출력
}

void evaluate_performance(int* labels, double** outputs, int num_samples) {
    int tp[NUM_CLASSES] = { 0 }, fp[NUM_CLASSES] = { 0 }, fn[NUM_CLASSES] = { 0 }, correct = 0;

    for (int i = 0; i < num_samples; i++) {
        int predicted_label = 0;
        for (int j = 1; j < OUTPUT_NODES; j++) {
            if (outputs[i][j] > outputs[i][predicted_label]) {
                predicted_label = j;
            }
        }

        if (labels[i] < 0 || labels[i] >= NUM_CLASSES) {
            printf("Invalid label [%d] at index %d\n", labels[i], i);
            continue;
        }

        if (predicted_label == labels[i]) {
            tp[predicted_label]++;
            correct++;
        }
        else {
            fp[predicted_label]++;
            fn[labels[i]]++;
        }
    }

    double total_precision = 0.0, total_recall = 0.0;

    printf("\nPerformance Metrics:\n");
    printf("Accuracy: %.2f%%\n", (double)correct / num_samples * 100);
    for (int i = 0; i < NUM_CLASSES; i++) {
        double precision = tp[i] > 0 ? (double)tp[i] / (tp[i] + fp[i]) : 0;
        double recall = tp[i] > 0 ? (double)tp[i] / (tp[i] + fn[i]) : 0;
        total_precision += precision;
        total_recall += recall;

        wprintf(L"%s: Precision = %.2f%%, Recall = %.2f%%\n", class_names[i], precision * 100, recall * 100);
    }

    // 평균 정밀도와 평균 재현율 계산
    double avg_precision = total_precision / NUM_CLASSES;
    double avg_recall = total_recall / NUM_CLASSES;

    printf("\nAverage Precision: %.2f%%\n", avg_precision * 100);
    printf("Average Recall: %.2f%%\n", avg_recall * 100);
}


int main() {
    int labels[NUM_SAMPLES];
    double* inputs[NUM_SAMPLES];
    double* outputs[NUM_SAMPLES];
    double* weights[HIDDEN_LAYERS + 1], * biases[HIDDEN_LAYERS + 1];

    // 가중치와 편향 로드
    load_weights_and_biases(weights, biases, "C:\\weights\\weights.bin", "C:\\biases\\biases.bin");

    // mean과 std 로드
    double* mean = malloc(INPUT_NODES * sizeof(double));
    double* std = malloc(INPUT_NODES * sizeof(double));
    load_mean_std(mean, std, "C:\\weights\\mean.bin", "C:\\weights\\std.bin");

    int num_samples = 0;
    load_test_data(DIRECTORY, inputs, labels, &num_samples);

    // 정규화 적용
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            inputs[i][j] = (inputs[i][j] - mean[j]) / std[j];
        }
    }

    // 정규화 후 모델 평가
    for (int i = 0; i < num_samples; i++) {
        outputs[i] = malloc(OUTPUT_NODES * sizeof(double));
        forward(inputs[i], weights, biases, outputs[i]);
    }

    evaluate_performance(labels, outputs, num_samples);

    // 메모리 해제
    for (int i = 0; i < num_samples; i++) {
        free(inputs[i]);
        free(outputs[i]);
    }
    for (int i = 0; i < HIDDEN_LAYERS + 1; i++) {
        free(weights[i]);
        free(biases[i]);
    }
    free(mean);
    free(std);

    return 0;
}

