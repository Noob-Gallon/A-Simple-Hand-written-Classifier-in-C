#define _CRT_SECURE_NO_WARNINGS
#define PI 3.14159265358979323846

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "model_function.h"

void load_target(int (*target)[TARGET_ALPHABET_NUM], int data_index, char label) {
    /*
        �־��� Label�� ���� target_vector�� �ʱ�ȭ�ϴ� �Լ�
    */

    // target���� ������ index�� ����ϴ� ����
    int target_point;

    // target�� ������ ���� ���·� ���ǵȴ�.
    // t : [ 1, 0, 0, 0, 0, 0, 0 ]
    // u : [ 0, 1, 0, 0, 0, 0, 0 ]
    // v : [ 0, 0, 1, 0, 0, 0, 0 ]
    // w : [ 0, 0, 0, 1, 0, 0, 0 ]
    // x : [ 0, 0, 0, 0, 1, 0, 0 ]
    // y : [ 0, 0, 0, 0, 0, 1, 0 ]
    // z : [ 0, 0, 0, 0, 0, 0, 1 ]

    // data�� target label ���� ����, ��Ī�Ǵ� index�� �����Ѵ�.
    if (label == 't') {
        target_point = 0;
    }
    else if (label == 'u') {
        target_point = 1;
    }
    else if (label == 'v') {
        target_point = 2;
    }
    else if (label == 'w') {
        target_point = 3;
    }
    else if (label == 'x') {
        target_point = 4;
    }
    else if (label == 'y') {
        target_point = 5;
    }
    else if (label == 'z') {
        target_point = 6;
    }

    // data�� label�� ���� target label vector�� �����.
    for (int target_value_index = 0; target_value_index < TARGET_ALPHABET_NUM; target_value_index++) {
        if (target_value_index == target_point) {
            target[data_index][target_value_index] = 1;
            continue;
        }

        target[data_index][target_value_index] = 0;
    }
}

void copy_memory_int_1dim(int* dest, int* source, int len) {
    /*
        �޸𸮿� ����ִ� ���� ������ ���� �Լ�
        1���� int �迭�� ���ؼ� �����Ѵ�.
    */

    for (int i = 0; i < len; i++) {
        dest[i] = source[i];
    }
}

void copy_memory_float_1dim(float* dest, float* source, int len) {
    /*
        �޸𸮿� ����ִ� ���� ������ ���� �Լ�
        1���� float �迭�� ���ؼ� �����Ѵ�.
    */

    for (int i = 0; i < len; i++) {
        dest[i] = source[i];
    }
}

void load_data_from_csv(
    float (*data_matrix)[DATA_SIZE], int (*target)[TARGET_ALPHABET_NUM], 
    char* path, int data_index
) {
    /*
        csv ���Ͽ��� pixel ����� label�� load�ϴ� �Լ�
        ������ path�� �ϳ��� .csv file�� �ǹ��ϹǷ�,
        �ϳ��� .csv file���� pixel���� label�� load�Ѵ�.
    */

    FILE* file = fopen(path, "r");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // * Buffer�� Size�� 1025�� ���� *
    // .csv�� ���Ե� value���� 0~255 ������ ���̴�.
    // ��� pixel value�� �� �ڸ� ����� �ϰ�, delimeter�� ,(comma)�̴�.
    // �� ��, pixel value(�� �ڸ� ��) + delimeter(�� �ڸ� ��)�� 4�̴�.
    // �׷��� data�� 16*16 �̹����̹Ƿ� �̷��� ���� 256�� ������ �� �ִ�.
    // ���� 4*256 = 1024�̰�, �� ���� �� ���� Label�� �����ϹǷ�,
    // Buffer�� ũ��� 1024+1�� 1025�� �����Ѵ�.
    char buffer[BUFFER_SIZE];

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        int buffer_index = 0;

        for (int pixel_value_index = 0; pixel_value_index < DATA_SIZE; pixel_value_index++) {
            // �ϳ��� pixel���� �ִ� �� �ڸ��̹Ƿ�, ũ�⸦ �ִ� 4�� �����Ѵ�.
            // ���ڸ����� NULL�� ����, ��Ȯ�ϰ� ���� ������ �������´�.
            char pixel_buffer[4];
            int pixel_buffer_index = 0;

            while (true) {
                char c = buffer[buffer_index];

                // ���� ","�� ���� ���, �����͸� �߰����� �ʰ� 
                // break�� �����Ͽ�, ���� �����͸� ������ �� �ֵ��� �Ѵ�.
                if (c == ',') {
                    buffer_index++;
                    pixel_buffer[pixel_buffer_index] = '\0';
                    break;
                }

                // pixel���� �����ϴ� character �ϳ��� ����
                pixel_buffer[pixel_buffer_index] = c;
                buffer_index++;
                pixel_buffer_index++;
            }

            // ������ pixel_value(char)�� type�� ��ȯ�Ͽ� �����Ѵ�.
            float pixel_value = atof(pixel_buffer);

            // ������ ��ó��
            // pixel value�� data�� ����ϹǷ�, �����ϰ� 255�� ������
            // ���� ũ�⿡ ���� ������� ���ҽ�Ű��, ǥ������ ������Ų��.
            data_matrix[data_index-1][pixel_value_index] = pixel_value/255;
        }

        // ��� pixel value�� ������ parse �Ͽ����Ƿ�, label�� �����Ѵ�.
        // ���� label�� ��ġ�� buffer_index�̴�.
        char label = buffer[buffer_index];
        load_target(target, data_index-1, label);
    }

    // ���� �ݱ�
    fclose(file);
}

void get_file_name_by_index(char* path_name, int data_index, int path_name_len) {
    /*
        ���޹��� file_name �ڿ� ���� �̸��� �߰��ϴ� �Լ�. 
        ex) 1.csv, 2.csv, 3.csv ���� path_prefix �ڿ� ���δ�.
    */

    char data_index_str[5]; // �ϳ��� pixel value�� �о�� ������ array
    char path_postfix[] = ".csv"; // data�� path�� ���� ���� postfix

    // data_index�� string���� ��ȯ�Ѵ�.
    sprintf(data_index_str, "%d", data_index);

    // �������� ��ȣ�� ����(���� �̸��� ��ȣ�� ��������.)
    int data_index_str_len = strlen(data_index_str);

    // path_name�� ���� prefix�� ���� (.csv)
    int path_postfix_len = strlen(path_postfix);

    // path_name�� data_index�� �߰��Ѵ�.
    for (int idx = 0; idx < data_index_str_len; idx++) {
        path_name[path_name_len + idx] = data_index_str[idx];
    }

    // path_name�� data_index_str�� �߰������Ƿ�,
    // path_name_len���� data_index_str_len�� �����ش�.
    path_name_len += data_index_str_len;

    // path_name�� path_postfix�� �߰��Ѵ�.
    for (int idx = 0; idx < path_postfix_len; idx++) {
        path_name[path_name_len + idx] = path_postfix[idx];
    }
 
    // path_name�� path_postfix�� �߰������Ƿ�,
    // path_name_len���� path_postfix_len�� �����ش�.
    path_name_len += path_postfix_len;

    // String���μ� ����ϱ� ����, �������� NULL�� �߰��Ѵ�.
    path_name[path_name_len] = '\0';
}

void load_train_data(float (*train_data)[DATA_SIZE], int (*train_target)[TARGET_ALPHABET_NUM]) {
    /*
        train data(pixel + label)�� load�ϴ� �Լ�
    */
    
    char path_name[20]; // data�� path ��ü�� ������ character array
    char path_prefix[] = "./train/*"; // data�� path�� ù �κ��� �����ϴ� prefix

    int idx = 0;

    // path_name�� path_prefix�� �����Ѵ�.
    while (true) {
        char c = path_prefix[idx];

        // *�� �������, prefix�� ���̶�� �����Ѵ�.
        if (c == '*') {
            break;
        }

        path_name[idx] = c;
        idx++;
    }

    // idx�� path_name�� ���̷μ� ����� �� �ִ�.
    int path_name_len = idx;

    // ��ü train_data�� load�ϰ�, �����͸� �����Ѵ�.
    for (int data_index = 1; data_index <= TRAIN_DATA_NUM; data_index++) {
       // data�� index�� ���� file�� name string�� �����Ѵ�.
        get_file_name_by_index(path_name, data_index, path_name_len); 

        // data�� index�� ���� file���� pixel value�� target label vector�� �����Ѵ�.
        load_data_from_csv(train_data, train_target, path_name, data_index);
    }
}

void load_test_data(float (*test_data)[DATA_SIZE], int (*test_target)[TARGET_ALPHABET_NUM]) {
    /*
        test data(pixel+label)�� load�ϴ� �Լ�. ������ load_train_data�� �����ϴ�.
    */

    char path_name[20];
    char path_prefix[] = "./test/*";

    int idx = 0;

    // path_name�� path_prefix�� �����Ѵ�.
    while (true) {
        char c = path_prefix[idx];

        // *�� �������, prefix�� ���̶�� �����Ѵ�.
        if (c == '*') {
            break;
        }

        path_name[idx] = c;
        idx++;
    }

    // idx�� path_name�� ���̷μ� ����� �� �ִ�.
    int path_name_len = idx;

    // ��ü test_data�� load�ϰ�, �����͸� �����Ѵ�.
    for (int data_index = 1; data_index <= TEST_DATA_NUM; data_index++) {
        get_file_name_by_index(path_name, data_index, path_name_len);
        load_data_from_csv(test_data, test_target, path_name, data_index);
    }
}

void load_data(
    float(*train_data)[DATA_SIZE], int(*train_target)[TARGET_ALPHABET_NUM], 
    float(*test_data)[DATA_SIZE], int(*test_target)[TARGET_ALPHABET_NUM]
) {
    /*
        csv file�� �ε��Ͽ� train/test �����͸� ��������,
        pixel value�� label�� �и��Ͽ� �����ϴ� �Լ�
    */

    load_train_data(train_data, train_target); // train data �ҷ�����
    load_test_data(test_data, test_target); // test data �ҷ�����
}

void initialize_layer_node_num(int* arr, int* value) {
    /*
        layer ���� node ������ �ʱ�ȭ�ϴ� �Լ�
    */

    for (int layer_idx = 0; layer_idx < LAYER_NUM; layer_idx++) {
        arr[layer_idx] = value[layer_idx];
    }
}

void malloc_1dim_float_vector(float** vector, int vector_size) {
    /*
        ���޹��� float�� �����͸� �־��� ũ��� �����Ҵ��ϴ� �Լ� (utility function)
    */

    (*vector) = (float*)malloc(sizeof(float) * vector_size);
}

void malloc_2dim_float_matrix(float*** matrix, int row_size, int col_size) {
    /*
        ���޹��� float�� ������ �����͸� �־��� ũ��� �����Ҵ��ϴ� �Լ� (utility function)
    */

    *matrix = (float**)malloc(sizeof(float*) * row_size);
    
    for (int row = 0; row < row_size; row++) {
        (*matrix)[row] = (float*)malloc(sizeof(float) * col_size);
    }
}

void initialize_layer(Layer* layers, int* layer_node_num) {
    /*
        layer ���� �⺻���� �������� �ʱ�ȭ�ϴ� �Լ�
    */

    for (int layer_idx = 0; layer_idx < LAYER_NUM; layer_idx++) {
        // output layer�� layer_node_num�� �����ϸ�,
        // layer ��ü ��ü�� ���� index�� layer_node_num�� index��
        // layer ��ü�� index�� ���߱� ���� �뵵�̹Ƿ� �ʱ�ȭ�� �����Ѵ�.
        if (layer_idx == 0) {
            continue;
        }

        // layer�� layer_idx �Ҵ�
        layers[layer_idx].layer_idx = layer_idx;

        // layer�� activation type�� ����
        if (layer_idx == LAYER_NUM - 1) {
            layers[layer_idx].activation_type = SOFTMAX;
        }
        else {
            layers[layer_idx].activation_type = RELU;
        }

        // forward_pass���� ���� ������ cache �迭�� �����Ҵ�
        // �ʱ�ȭ�� forward pass �������� �̷������.
        layers[layer_idx].cache_Z = (float*)malloc(sizeof(float) * layer_node_num[layer_idx]);
        layers[layer_idx].cache_prev_A = (float*)malloc(sizeof(float) * layer_node_num[layer_idx - 1]);
    }
}

float get_random_number(float std) {
    /*
        ����� 0�̰� ǥ�������� std�� ���콺 ������ 
        ������ �������� �ϳ� ��ȯ�ϴ� �Լ�
    */

    // Box-Muller �˰����� ����Ͽ� random ������ �����Ѵ�.

    double u1, u2; // �� ���� �������� [0,1) ������ ����
    double z;      // ǥ�� ���� ������ ������ ����

    // Box-Muller ��ȯ �˰����� ����Ͽ� ǥ�� ���� ������ ������ ���� ����
    u1 = rand() / (RAND_MAX + 1.0);
    u2 = rand() / (RAND_MAX + 1.0);
    z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);

    // ����� 0�̰� ǥ�� ������ stddev�� ����þ� ������ �������� ����
    float random_number = std * z;
    
    return random_number;
}

void initialize_parameter(Layer* layers, int* layer_node_num) {
    /*
        layer ���� parameter�� �ʱ�ȭ�ϴ� �Լ�
    */

    // 0�� index�� layer�� input node�μ�, �������� ����ġ�� ���� �ʴ´�.
    // ���� layer�� node num�� ����ϰ�, �ʱ�ȭ�� 1������ �����Ѵ�.
    for (int layer_idx = 1; layer_idx < LAYER_NUM; layer_idx++) {

        // weight�� ���� �����Ҵ�
        malloc_2dim_float_matrix(
            &(layers[layer_idx].weight), 
            layer_node_num[layer_idx], 
            layer_node_num[layer_idx-1]
        );

        // dW�� ���� �����Ҵ�
        malloc_2dim_float_matrix(
            &(layers[layer_idx].dW),
            layer_node_num[layer_idx],
            layer_node_num[layer_idx - 1]
        );

        // bias�� ���� �����Ҵ�
        malloc_1dim_float_vector(
            &(layers[layer_idx].bias),
            layer_node_num[layer_idx]
        );

        // db�� ���� �����Ҵ�
        malloc_1dim_float_vector(
            &(layers[layer_idx].db),
            layer_node_num[layer_idx]
        );
    }
    // ----- weight/dw, bias/db�� ���� �����Ҵ� �Ϸ� -----

    // random �Լ��� ���� seed�� ����
    srand(1);

    // �����Ҵ��� �Ϸ�Ǿ����Ƿ�, ���� ���� ä���ִ´�.
    // hidden layer�� ReLU�� ����ϹǷ�, He Initialization�� ����Ѵ�.
    // output layer�� Softmax�� ����ϹǷ�, Xavier Initialization�� ����Ѵ�.
    for (int layer_idx = 1; layer_idx < LAYER_NUM; layer_idx++) {
        float std; // ǥ������
        float random_number; // random ����

        // 1. weight/dW initialization
        // output layer�� weight/dW �ʱ�ȭ (Xavier)
        // ��, �����δ� Xavier�� �ƴ� LeCun ���������, ������ ���翡�� Xavier ����̶�� �����־�
        // �� ������Ʈ������ LeCun�� Xavier��� ����Ѵ�.
        if (layer_idx == LAYER_NUM - 1) {
            std = sqrt(1.0 / layer_node_num[layer_idx - 1]); // Xavier ǥ������

            for (int i = 0; i < layer_node_num[layer_idx]; i++) { // ���� layer�� ���̸�ŭ
                for (int j = 0; j < layer_node_num[layer_idx - 1]; j++) { // ���� layer�� ���̸�ŭ
                    random_number = get_random_number(std);
                    layers[layer_idx].weight[i][j] = random_number; // weight �߰�
                    layers[layer_idx].dW[i][j] = 0; // dW�� gradient�� �����ϹǷ�, 0���� �ʱ�ȭ
                }
            }
        }
        // hidden layer�� weight/dW �ʱ�ȭ (He)
        else {
            std = sqrt(2.0 / layer_node_num[layer_idx - 1]); // He ǥ������

            for (int i = 0; i < layer_node_num[layer_idx]; i++) { // ���� layer�� ���̸�ŭ
                for (int j = 0; j < layer_node_num[layer_idx - 1]; j++) { // ���� layer�� ���̸�ŭ
                    random_number = get_random_number(std);
                    layers[layer_idx].weight[i][j] = random_number; // weight �߰�
                    layers[layer_idx].dW[i][j] = 0; // dW�� gradient�� �����ϹǷ�, 0���� �ʱ�ȭ
                }
            }
        }

        // 2. bias/db initialization
        // bias�� hidden/output�� ������� ��� 0���� �ʱ�ȭ�Ѵ�.
        for (int i = 0; i < layer_node_num[layer_idx]; i++) { // ���� layer�� ���̸�ŭ
            layers[layer_idx].bias[i] = 0;
            layers[layer_idx].db[i] = 0;
        }
    }
}

void free_allocated_memory(Layer* layers, int* layer_node_num) {
    /*
        �н� �������� ����� ��� �����Ҵ� �޸𸮸� �����ϴ� �Լ�
    */

    for (int layer_idx = 1; layer_idx < LAYER_NUM; layer_idx++) {
        
        // free weight matrix
        for (int row = 0; row < layer_node_num[layer_idx]; row++) {
            free(layers[layer_idx].weight[row]);
        }
        free(layers[layer_idx].weight);

        // free dW
        for (int row = 0; row < layer_node_num[layer_idx]; row++) {
            free(layers[layer_idx].dW[row]);
        }
        free(layers[layer_idx].dW);

        // free bias
        free(layers[layer_idx].bias);

        // free db
        free(layers[layer_idx].db);

        // free cache_Z
        free(layers[layer_idx].cache_Z);

        // free cache_prev_A
        free(layers[layer_idx].cache_prev_A);
    }
}

float relu(float value) {
    /*
        activation function ReLU
    */

    if (value > 0) {
        return value;
    }
    else {
        return 0;
    }
}

void element_wise_relu(float* activation, int current_layer_node_num) {
    /*
        ReLU�� ���� ���� ���� ���� element wise�� �����ϴ� �Լ�
    */

    for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
        activation[node_index] = relu(activation[node_index]);
    }
}

float max(float* activation) {
    /*
        ���޹��� float vector(output)���� ���� ū ���� ã�� ��ȯ�ϴ� �Լ�
    */

    float max_value = activation[0];

    for (int node_index = 1; node_index < TARGET_ALPHABET_NUM; node_index++) {
        if (activation[node_index] > max_value) {
            max_value = activation[node_index];
        }
    }

    return max_value;
}

int max_index(float* activation) {
    /*
        ���޹��� float vector(output)���� ���� ū ���� index�� ��ȯ�ϴ� �Լ�
    */

    int max_value_idx = 0;

    for (int node_index = 1; node_index < TARGET_ALPHABET_NUM; node_index++) {
        if (activation[node_index] > activation[max_value_idx]) {
            max_value_idx = node_index;
        }
    }

    return max_value_idx;
}

void softmax(float* activation, int current_layer_node_num) {
    /*
        activation function Softmax
    */
    
    float total_exp_value_sum = 0; // activation�� �����ϴ� ��� value�� exp�� ���� ���� ��
    float max_value = max(activation); // activation���� ���� ū ���� ���Ѵ�.

    for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
        // activation���� max_value�� �� �Ϳ� exp�� ���� ���� ���Ѵ�.
        double value = activation[node_index] - max_value;
        float exp_value = exp(value);

        // �ش� ���� total_exp_value_sum�� ���Ѵ�.
        total_exp_value_sum += exp_value;

        // activation[node_index]�� exp_value�� �ٲ۴�.
        // ���� for loop���� total_exp_value_sum���� ������ ���̴�.
        activation[node_index] = exp_value;
    }

    // activation�� �� value�� total_exp_value_sum���� �����ش�.
    for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
        activation[0] /= total_exp_value_sum;
    }
}

void dot_product(float* activation, int prev_layer_node_num, int current_layer_node_num, float** weight) {
    // activation�� ������ �ӽ� vector ����
    float* output = (float*)malloc(sizeof(float) * current_layer_node_num);

    // dot product ����
    for (int i = 0; i < current_layer_node_num; i++) {
        float value = 0;

        for (int j = 0; j < prev_layer_node_num; j++) {
            value += weight[i][j] * activation[j];
        }

        output[i] = value;
    }

    // output�� ���� activation�� �����Ѵ�.
    copy_memory_float_1dim(activation, output, current_layer_node_num);

    // �����Ҵ� �޸� ����
    free(output);
}

void forward_pass(float* activation, int* layer_node_num, float (*data)[DATA_SIZE], int data_index, Layer* layers) {
    // ���� data�� ���� activation���� �����Ѵ�. (�ʱ� input��)
    
    // data���� activation���� ���� �����Ѵ�.
    for (int i = 0; i < DATA_SIZE; i++) {
        activation[i] = data[data_index][i];
    }
    
    // for loop���� ���� ���̾��� ����� ������ ����
    float prev_activation[DATA_SIZE];

    for (int layer_index = 1; layer_index < LAYER_NUM; layer_index++) {
        // prev_activation�� activation�� ���� �����Ͽ� �����Ѵ�.
        // loop�� �ݺ��ϸ�, DATA_SIZE���� ���� ���Ǵ� ������ �پ��� ����.
        copy_memory_float_1dim(prev_activation, activation, layer_node_num[layer_index - 1]);

        int prev_layer_node_num = layer_node_num[layer_index-1];
        int current_layer_node_num = layer_node_num[layer_index];
        
        // dot product�� �����Ѵ�. �Լ� ������ ������, 
        // ���� activation���� ���ο� layer�� Z���� ����ִ�.
        dot_product(activation, prev_layer_node_num, current_layer_node_num, layers[layer_index].weight);
        
        // Z�� ���� bias�� ���� �����ش�.
        for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
            activation[node_index] += layers[layer_index].bias[node_index];
        }

        // ��������� activation function�� ��ġ�� �ʾ����Ƿ�,
        // activation ������ Z���� ������ �ִ� ��Ȳ�̴�.
        // ���� activation�� ���� cache_Z�� �����Ѵ�.
        copy_memory_float_1dim(layers[layer_index].cache_Z, activation, current_layer_node_num);

        // ���� prev_A�� ���� cache�� �����Ѵ�.
        copy_memory_float_1dim(layers[layer_index].cache_prev_A, prev_activation, prev_layer_node_num);

        // activation function ����
        // ���� Layer�� activation function�� �Ǵ��Ͽ� activation�� �����Ѵ�.
        if (layers[layer_index].activation_type == SOFTMAX) {
            softmax(activation, current_layer_node_num);
        }
        else if (layers[layer_index].activation_type == RELU) {
            element_wise_relu(activation, current_layer_node_num);
        }
    }
}

void calc_output_layer_dZ(float* dZ, int* current_target, float* activation) {
    /*
        Softmax�� Activation Function���� ����ϴ�
        Output Layer�� ���� dZ�� ���Ѵ�.
        (dZ�� Chain Rule �������� Loss�� Z�� �̺��� ���� �ǹ��Ѵ�.)
    */

    for (int node_idx = 0; node_idx < TARGET_ALPHABET_NUM; node_idx++) {
        dZ[node_idx] = activation[node_idx] - current_target[node_idx];
    }
}

void calc_hidden_layer_dZ(float* dZ, float* dA, int current_layer_idx, int current_layer_node_num, Layer* layers) {
    /*
        ReLU�� Activation Function���� ����ϴ�
        Output Layer�� ���� dZ�� ���Ѵ�.
        (dZ�� Chain Rule ���������� Loss�� Z�� �̺��� ���� �ǹ��Ѵ�.)
    */

    // cache�κ��� ������ ������ �� Z���� �����´�.
    float* Z = (float*)malloc(sizeof(float)*current_layer_node_num);
    copy_memory_float_1dim(Z, layers[current_layer_idx].cache_Z, current_layer_node_num);
    
    // Z�� size�� (curent_layer_node_num, 1)�̴�. ���� ������ ���� for loop�� ����,
    // ReLU�� gradient�� ���� ������ 1, 0�Ǵ� ������� 0�̹Ƿ� �Ʒ��� ���� Z�� ���� ��ȯ�Ѵ�.
    for (int node_idx = 0; node_idx < current_layer_node_num; node_idx++) {
        if (Z[node_idx] < 0) {
            Z[node_idx] = 0;
        }
        else {
            Z[node_idx] = 1;
        }
    }

    // ������ ���� ReLU�� �̺а��� dA�� ���ؼ� dZ�� ���Ѵ�. 
    for (int node_idx = 0; node_idx < current_layer_node_num; node_idx++) {
        dZ[node_idx] = dA[node_idx] * Z[node_idx];
    }

    // cache�� ���� �������� ���� ����ߴ� Z�� �޸𸮸� �����Ѵ�.
    free(Z);
}

void calc_dW(float* dZ, int current_layer_idx, int current_layer_node_num, int prev_layer_node_num, Layer* layers) {
    /*
        ���ڷ� ���޹��� ���� ����� layer�� dW�� ������Ʈ �ϴ� �Լ�
    */

    // ������ ���ؼ� dW�� ������Ʈ ����
    for (int i = 0; i < current_layer_node_num; i++) {
        for (int j = 0; j < prev_layer_node_num; j++) {
            float dW_value = dZ[i] * layers[current_layer_idx].cache_prev_A[j];
            layers[current_layer_idx].dW[i][j] += dW_value;
        }
    }
}
void calc_db(float* dZ, int current_layer_idx, int current_layer_node_num, Layer* layers) {
    /*
        ���ڷ� ���޹��� ���� ����� layer�� db�� ������Ʈ �ϴ� �Լ�
    */

    // db�� dZ�� ���� �Ȱ����Ƿ�, �״�� ���Ѵ�.
    for (int i = 0; i < current_layer_node_num; i++) {
        layers[current_layer_idx].db[i] += dZ[i];
    }
}

void calc_dA_prev(float* dA_prev, float* dZ, int current_layer_idx, 
    int current_layer_node_num, int prev_layer_node_num, Layer* layers) {
    /*
        ���� ���̾�� ������ dA_prev�� ���� ����ϴ� �Լ�
    */

    for (int i = 0; i < prev_layer_node_num; i++) {
        float dA_prev_value = 0;

        for (int j = 0; j < current_layer_node_num; j++) {
            dA_prev_value += layers[current_layer_idx].weight[j][i] * dZ[j];
        }

        dA_prev[i] = dA_prev_value;
    }
}

void calc_softmax_gradient(float* dA_prev, float* activation, int* layer_node_num, 
    int(*train_target)[TARGET_ALPHABET_NUM], int data_index, Layer* layers) {
    /*
        softmax layer�� output layer�̹Ƿ�, forward pass������ ���� activation(output)�� ���޹޾�
        gradient�� ����� layer�� gradient ���� ������Ʈ�Ѵ�.
    */

    // ���� train_data�� ���� target ���� �����Ͽ� �����´�.
    int current_target[TARGET_ALPHABET_NUM];
    copy_memory_int_1dim(current_target, train_target[data_index], TARGET_ALPHABET_NUM);

    // Softmax�� Output layer�� ���ȴٰ� �����ϱ� ������,
    // layer�� index�� LAYER_NUM-1�� �����ϸ�,
    // node�� ������ TARGET_ALPHABET_NUM���� �����Ѵ�.
    int current_layer_idx = LAYER_NUM - 1;
    int current_layer_node_num = TARGET_ALPHABET_NUM;
    
    int prev_layer_idx = current_layer_idx - 1;
    int prev_layer_node_num = layer_node_num[prev_layer_idx];

    // output layer�̹Ƿ�, dZ�� ũ��� TARGET_ALPHABET_NUM�̴�.
    float dZ[TARGET_ALPHABET_NUM];

    // dW�� db�� ����Ѵ�.
    // 1) dW�� db�� ����ϱ� ���ؼ��� dZ�� �ʿ��ϹǷ�, ���� dZ�� ����Ѵ�.
    // 2) dZ�� prev_activation�� �������� dW�� ����Ѵ�.
    // 3) dZ�� �������� db�� ����Ѵ�.
    calc_output_layer_dZ(dZ, current_target, activation); // dZ vector�� ���� ����� ����. (index 0~6)
    calc_dW(dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers); // dW�� ����ϰ� ������Ʈ
    calc_db(dZ, current_layer_idx, current_layer_node_num, layers); // db�� ����ϰ� ������Ʈ
    
    // dZ�� Weight�� �̿��� ���� Layer�� ������ �� dA_prev(prev_activation�� gradient)�� ����Ѵ�.
    calc_dA_prev(dA_prev, dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers);
}

void calc_relu_gradient(float* dA, int* layer_node_num, int current_layer_idx, Layer* layers) {
    /*
        ReLU�� ���� Gradient�� ����ϴ� �Լ�
        ���� �𵨿����� Hidden Layer�� Activation Function���� ReLU�� ����ϹǷ�,
        Ư�� Hidden Layer�� ���� Gradient�� ����ϴ� ������ ������ �� �ִ�.
    */

    int current_layer_node_num = layer_node_num[current_layer_idx];

    int prev_layer_idx= current_layer_idx - 1;
    int prev_layer_node_num = layer_node_num[prev_layer_idx];

    // dZ���� �����ϱ� ���� vector�� �����Ҵ��Ѵ�.
    float* dZ = (float*)malloc(sizeof(float) * current_layer_node_num);

    // dW�� db�� ����Ѵ�.
    // 1) dW�� db�� ����ϱ� ���ؼ��� dZ�� �ʿ��ϹǷ�, ���� dZ�� ����Ѵ�.
    // 2) dZ�� prev_activation�� �������� dW�� ����Ѵ�.
    // 3) dZ�� �������� db�� ����Ѵ�.
    calc_hidden_layer_dZ(dZ, dA, current_layer_idx, current_layer_node_num, layers); // dZ vector�� ���� ����� ����
    calc_dW(dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers); // dW�� ����ϰ� ������Ʈ
    calc_db(dZ, current_layer_idx, current_layer_node_num, layers); // db�� ����ϰ� ������Ʈ
    calc_dA_prev(dA, dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers); // ���� ���̾�� ���� dA_prev ���

    // �ӽ÷� Ȱ���� �޸� ����
    free(dZ);
}

void backward_pass(float* activation, int* layer_node_num, int(*train_target)[TARGET_ALPHABET_NUM], int data_index, Layer* layers) {
    /*
        backward pass�� ��ü ������ �����ϴ� �Լ�
        �� ���� example�� ���� backward pass�� �����Ѵ�.
    */

    // activation�� output ���� backward pass�� �Ѱ��ִ� �����̴�.
    // ���� ���� �����ϰ� �Ѱ���ٸ�, �� ������ �����ٰ� �� �� �ִ�.

    // backward_pass���� ���ʿ� current_layer_idx�� output_layer_idx�� ����.
    int current_layer_idx = LAYER_NUM - 1;
    int current_layer_node_num = layer_node_num[current_layer_idx];
    
    int prev_layer_idx = current_layer_idx - 1;
    int prev_layer_node_num = layer_node_num[prev_layer_idx];

    // gardient�� ���ϴ� ��������, dA_prev�� ����Ͽ�
    // �����͸� �����ϴ� ������ �ϴ� vector.
    // dA_prev�� �ִ� ũ��� DATA_SIZE�̹Ƿ� size�� �̿� ���� �����Ѵ�.
    float dA_prev[DATA_SIZE];

    // calc_softmax_gradient�� ��ġ�� output layer�� ���� dW/db�� ����ǰ�,
    // dA_prev�� ���� hidden layer�� �Ѱ��� ���� ����ȴ�.
    calc_softmax_gradient(dA_prev, activation, layer_node_num, train_target, data_index, layers);

    for (int layer_idx = (LAYER_NUM - 1) - 1; layer_idx > 0; layer_idx--) {
        // hidden layer�� ���� gradient�� ����Ѵ�.
        current_layer_idx = layer_idx;

        // ���� layer�� ���ؼ� gradient�� ��������, dA_prev�� ���� ����ȴ�.
        // ����� dA_prev ���� ������ loop���� prev layer�� activation gradient�� ���ȴ�.
        calc_relu_gradient(dA_prev, layer_node_num, current_layer_idx, layers);
    }
}

void update_parameter(Layer* layers, int* layer_node_num) {
    /*
        ����� gradient���� �������� parameter�� ������Ʈ�ϰ�,
        ���� ��� 0���� �ʱ�ȭ�ϴ� �Լ�
    */

    for (int layer_idx = 1; layer_idx < LAYER_NUM; layer_idx++) {
        int current_layer_idx = layer_idx;
        int prev_layer_idx = current_layer_idx - 1;

        for (int i = 0; i < layer_node_num[current_layer_idx]; i++) {
            for (int j = 0; j < layer_node_num[prev_layer_idx]; j++) {
                layers[current_layer_idx].dW[i][j] /= TRAIN_DATA_NUM;
                layers[current_layer_idx].weight[i][j] -= LEARNING_RATE * layers[current_layer_idx].dW[i][j];
                layers[current_layer_idx].dW[i][j] = 0;
            }

            layers[current_layer_idx].db[i] /= TRAIN_DATA_NUM;
            layers[current_layer_idx].bias[i] -= LEARNING_RATE * layers[current_layer_idx].db[i];
            layers[current_layer_idx].db[i] = 0;
        }
    }
}

bool isCorrect(float* activation, int* layer_node_num, int (*train_target)[TARGET_ALPHABET_NUM], int train_data_index) {
    /*
        forward pass�� ����� �������� Ȯ���ϴ� �Լ�
    */

    int predict_index = max_index(activation);
    int target_index = NULL;

    for (int idx = 0; idx < TARGET_ALPHABET_NUM; idx++) {
        if (train_target[train_data_index][idx] == 1) {
            target_index = idx;
        }
    }

    if (predict_index == target_index) {
        return true;
    }
    else {
        return false;
    }
}

void train(int* layer_node_num, float (*train_data)[DATA_SIZE], int (*train_target)[TARGET_ALPHABET_NUM], Layer* layers) {
    /*
        Model training�� ��ü ������ �����ϴ� �Լ�
    */

    // Accuracy ����� ���� float type�� train data set�� ���� ���� ����
    float total_train_data_num = 420.0;

    // train ����, ������ epoch ����ŭ for loop�� �����Ѵ�.
    for (int epoch = 0; epoch < EPOCH_NUM; epoch++) {
        printf("epoch : %d\n", epoch);
        
        // Accuracy ����� ���� ���� ������ ���� ī���� ���� ����
        float correct = 0.0;

        // ������ train data ������ŭ for loop ����. �� ���� epoch�� �ǹ��Ѵ�.
        for (int train_data_index = 0; train_data_index < TRAIN_DATA_NUM; train_data_index++) {

            // forward pass�� output�� �޾ƿ� activation vector ����
            // Layer�� ��� �ִ� ũ�Ⱑ DATA_SIZE�̹Ƿ�, 
            // activation vector�� ũ�⸦ DATA_SIZE�� �����.
            // ���Ŀ� backward_pass�� ���Ǹ� Layer�� ��� ������ ������ ������ 
            // indexing�� �������Ͽ� ���� ����Ѵ�.
            float activation[DATA_SIZE];

            // �� ���� example�� ���Ͽ� forward pass ����
            forward_pass(activation, layer_node_num, train_data, train_data_index, layers);

            // forward_pass�� ����� �������� Ȯ��
            if (isCorrect(activation, layer_node_num, train_target, train_data_index)) {
                correct += 1;
            }

            // �� ���� example�� ���Ͽ� backward pass ����
            backward_pass(activation, layer_node_num, train_target, train_data_index, layers);
        }

        // �� ���� epoch���� ���� ����� Accuracy�� ���
        printf("Accuracy : %f (%c)\n", (correct/total_train_data_num)*100.0, '%');
        
        // backward pass�� ���� ���� gradient�� ��������, 1 epoch�� ���� parameter update�� ����
        update_parameter(layers, layer_node_num);
    }
}

void predict(int* layer_node_num, float(*test_data)[DATA_SIZE], int(*test_target)[TARGET_ALPHABET_NUM], Layer* layers) {
    /*
        �־��� test data set�� ���ؼ� ����� �����ϴ� �Լ� (Test)
    */

    // Accuracy ����� ���� float type�� test data set�� ���� ���� ����
    float total_test_data_num = 140.0;

    // test ����
    float correct = 0.0;

    for (int test_data_index = 0; test_data_index < TEST_DATA_NUM; test_data_index++) {
        // forward pass�� output�� �޾ƿ� activation vector ����
        // Layer�� ��� �ִ� ũ�Ⱑ DATA_SIZE�̹Ƿ�, 
        // activation vector�� ũ�⸦ DATA_SIZE�� �����.
        // ���Ŀ� backward_pass�� ���Ǹ� Layer�� ��� ������ ������ ������ 
        // indexing�� �������Ͽ� ���� ����Ѵ�.
        float activation[DATA_SIZE];

        forward_pass(activation, layer_node_num, test_data, test_data_index, layers);

        if (isCorrect(activation, layer_node_num, test_target, test_data_index)) {
            correct += 1;
        }

        backward_pass(activation, layer_node_num, test_target, test_data_index, layers);
    }

    // ��� ���
    printf("\n\n\nTest Accuracy : %f (%c)\n", (correct / total_test_data_num) * 100.0, '%');
}