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
        주어진 Label에 따라 target_vector를 초기화하는 함수
    */

    // target에서 정답의 index를 기록하는 변수
    int target_point;

    // target은 다음과 같은 형태로 정의된다.
    // t : [ 1, 0, 0, 0, 0, 0, 0 ]
    // u : [ 0, 1, 0, 0, 0, 0, 0 ]
    // v : [ 0, 0, 1, 0, 0, 0, 0 ]
    // w : [ 0, 0, 0, 1, 0, 0, 0 ]
    // x : [ 0, 0, 0, 0, 1, 0, 0 ]
    // y : [ 0, 0, 0, 0, 0, 1, 0 ]
    // z : [ 0, 0, 0, 0, 0, 0, 1 ]

    // data의 target label 값에 따라서, 매칭되는 index를 지정한다.
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

    // data의 label에 따라서 target label vector를 만든다.
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
        메모리에 들어있는 값을 복사해 오는 함수
        1차원 int 배열에 대해서 동작한다.
    */

    for (int i = 0; i < len; i++) {
        dest[i] = source[i];
    }
}

void copy_memory_float_1dim(float* dest, float* source, int len) {
    /*
        메모리에 들어있는 값을 복사해 오는 함수
        1차원 float 배열에 대해서 동작한다.
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
        csv 파일에서 pixel 값들과 label을 load하는 함수
        지정된 path는 하나의 .csv file을 의미하므로,
        하나의 .csv file에서 pixel값과 label을 load한다.
    */

    FILE* file = fopen(path, "r");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // * Buffer의 Size가 1025인 이유 *
    // .csv에 포함된 value들은 0~255 사이의 값이다.
    // 모든 pixel value가 세 자리 수라고 하고, delimeter는 ,(comma)이다.
    // 이 때, pixel value(세 자리 수) + delimeter(한 자리 수)는 4이다.
    // 그런데 data가 16*16 이미지이므로 이러한 것이 256개 존재할 수 있다.
    // 따라서 4*256 = 1024이고, 맨 끝에 한 글자 Label이 존재하므로,
    // Buffer의 크기는 1024+1인 1025로 설정한다.
    char buffer[BUFFER_SIZE];

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        int buffer_index = 0;

        for (int pixel_value_index = 0; pixel_value_index < DATA_SIZE; pixel_value_index++) {
            // 하나의 pixel값은 최대 세 자리이므로, 크기를 최대 4로 설정한다.
            // 끝자리에는 NULL이 들어가서, 명확하게 숫자 영역을 구분짓는다.
            char pixel_buffer[4];
            int pixel_buffer_index = 0;

            while (true) {
                char c = buffer[buffer_index];

                // 만약 ","를 만날 경우, 데이터를 추가하지 않고 
                // break를 실행하여, 다음 데이터를 가져올 수 있도록 한다.
                if (c == ',') {
                    buffer_index++;
                    pixel_buffer[pixel_buffer_index] = '\0';
                    break;
                }

                // pixel값을 구성하는 character 하나를 저장
                pixel_buffer[pixel_buffer_index] = c;
                buffer_index++;
                pixel_buffer_index++;
            }

            // 가져온 pixel_value(char)의 type을 변환하여 저장한다.
            float pixel_value = atof(pixel_buffer);

            // 데이터 전처리
            // pixel value를 data로 사용하므로, 간단하게 255로 나눠서
            // 값에 크기에 의한 영향력을 감소시키고, 표현력을 증가시킨다.
            data_matrix[data_index-1][pixel_value_index] = pixel_value/255;
        }

        // 모든 pixel value를 위에서 parse 하였으므로, label을 저장한다.
        // 현재 label의 위치는 buffer_index이다.
        char label = buffer[buffer_index];
        load_target(target, data_index-1, label);
    }

    // 파일 닫기
    fclose(file);
}

void get_file_name_by_index(char* path_name, int data_index, int path_name_len) {
    /*
        전달받은 file_name 뒤에 파일 이름을 추가하는 함수. 
        ex) 1.csv, 2.csv, 3.csv 등을 path_prefix 뒤에 붙인다.
    */

    char data_index_str[5]; // 하나의 pixel value를 읽어내어 저장할 array
    char path_postfix[] = ".csv"; // data의 path의 끝에 붙일 postfix

    // data_index를 string으로 변환한다.
    sprintf(data_index_str, "%d", data_index);

    // 데이터의 번호의 길이(파일 이름을 번호로 지정했음.)
    int data_index_str_len = strlen(data_index_str);

    // path_name에 붙일 prefix의 길이 (.csv)
    int path_postfix_len = strlen(path_postfix);

    // path_name에 data_index를 추가한다.
    for (int idx = 0; idx < data_index_str_len; idx++) {
        path_name[path_name_len + idx] = data_index_str[idx];
    }

    // path_name에 data_index_str을 추가했으므로,
    // path_name_len에도 data_index_str_len을 더해준다.
    path_name_len += data_index_str_len;

    // path_name에 path_postfix를 추가한다.
    for (int idx = 0; idx < path_postfix_len; idx++) {
        path_name[path_name_len + idx] = path_postfix[idx];
    }
 
    // path_name에 path_postfix를 추가했으므로,
    // path_name_len에도 path_postfix_len을 더해준다.
    path_name_len += path_postfix_len;

    // String으로서 사용하기 위해, 마지막에 NULL을 추가한다.
    path_name[path_name_len] = '\0';
}

void load_train_data(float (*train_data)[DATA_SIZE], int (*train_target)[TARGET_ALPHABET_NUM]) {
    /*
        train data(pixel + label)를 load하는 함수
    */
    
    char path_name[20]; // data의 path 전체를 저장할 character array
    char path_prefix[] = "./train/*"; // data의 path의 첫 부분을 구성하는 prefix

    int idx = 0;

    // path_name에 path_prefix를 복사한다.
    while (true) {
        char c = path_prefix[idx];

        // *를 만날경우, prefix가 끝이라고 생각한다.
        if (c == '*') {
            break;
        }

        path_name[idx] = c;
        idx++;
    }

    // idx는 path_name의 길이로서 사용할 수 있다.
    int path_name_len = idx;

    // 전체 train_data를 load하고, 데이터를 저장한다.
    for (int data_index = 1; data_index <= TRAIN_DATA_NUM; data_index++) {
       // data의 index에 따라 file의 name string을 생성한다.
        get_file_name_by_index(path_name, data_index, path_name_len); 

        // data의 index에 따라 file에서 pixel value와 target label vector를 생성한다.
        load_data_from_csv(train_data, train_target, path_name, data_index);
    }
}

void load_test_data(float (*test_data)[DATA_SIZE], int (*test_target)[TARGET_ALPHABET_NUM]) {
    /*
        test data(pixel+label)를 load하는 함수. 역할은 load_train_data와 동일하다.
    */

    char path_name[20];
    char path_prefix[] = "./test/*";

    int idx = 0;

    // path_name에 path_prefix를 복사한다.
    while (true) {
        char c = path_prefix[idx];

        // *를 만날경우, prefix가 끝이라고 생각한다.
        if (c == '*') {
            break;
        }

        path_name[idx] = c;
        idx++;
    }

    // idx는 path_name의 길이로서 사용할 수 있다.
    int path_name_len = idx;

    // 전체 test_data를 load하고, 데이터를 저장한다.
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
        csv file을 로드하여 train/test 데이터를 가져오고,
        pixel value와 label을 분리하여 저장하는 함수
    */

    load_train_data(train_data, train_target); // train data 불러오기
    load_test_data(test_data, test_target); // test data 불러오기
}

void initialize_layer_node_num(int* arr, int* value) {
    /*
        layer 별로 node 개수를 초기화하는 함수
    */

    for (int layer_idx = 0; layer_idx < LAYER_NUM; layer_idx++) {
        arr[layer_idx] = value[layer_idx];
    }
}

void malloc_1dim_float_vector(float** vector, int vector_size) {
    /*
        전달받은 float형 포인터를 주어진 크기로 동적할당하는 함수 (utility function)
    */

    (*vector) = (float*)malloc(sizeof(float) * vector_size);
}

void malloc_2dim_float_matrix(float*** matrix, int row_size, int col_size) {
    /*
        전달받은 float형 이차원 포인터를 주어진 크기로 동적할당하는 함수 (utility function)
    */

    *matrix = (float**)malloc(sizeof(float*) * row_size);
    
    for (int row = 0; row < row_size; row++) {
        (*matrix)[row] = (float*)malloc(sizeof(float) * col_size);
    }
}

void initialize_layer(Layer* layers, int* layer_node_num) {
    /*
        layer 별로 기본적인 변수들을 초기화하는 함수
    */

    for (int layer_idx = 0; layer_idx < LAYER_NUM; layer_idx++) {
        // output layer는 layer_node_num을 제외하면,
        // layer 객체 자체는 그저 index를 layer_node_num의 index와
        // layer 객체의 index를 맞추기 위한 용도이므로 초기화를 생략한다.
        if (layer_idx == 0) {
            continue;
        }

        // layer에 layer_idx 할당
        layers[layer_idx].layer_idx = layer_idx;

        // layer의 activation type을 설정
        if (layer_idx == LAYER_NUM - 1) {
            layers[layer_idx].activation_type = SOFTMAX;
        }
        else {
            layers[layer_idx].activation_type = RELU;
        }

        // forward_pass에서 값을 저장할 cache 배열을 동적할당
        // 초기화는 forward pass 과정에서 이루어진다.
        layers[layer_idx].cache_Z = (float*)malloc(sizeof(float) * layer_node_num[layer_idx]);
        layers[layer_idx].cache_prev_A = (float*)malloc(sizeof(float) * layer_node_num[layer_idx - 1]);
    }
}

float get_random_number(float std) {
    /*
        평균이 0이고 표준편차가 std인 가우스 분포를 
        따르는 랜덤값을 하나 반환하는 함수
    */

    // Box-Muller 알고리즘을 사용하여 random 변수를 생성한다.

    double u1, u2; // 두 개의 독립적인 [0,1) 범위의 난수
    double z;      // 표준 정규 분포를 따르는 난수

    // Box-Muller 변환 알고리즘을 사용하여 표준 정규 분포를 따르는 난수 생성
    u1 = rand() / (RAND_MAX + 1.0);
    u2 = rand() / (RAND_MAX + 1.0);
    z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);

    // 평균이 0이고 표준 편차가 stddev인 가우시안 분포를 따르도록 조절
    float random_number = std * z;
    
    return random_number;
}

void initialize_parameter(Layer* layers, int* layer_node_num) {
    /*
        layer 별로 parameter를 초기화하는 함수
    */

    // 0번 index인 layer는 input node로서, 실질적인 가중치를 갖지 않는다.
    // 따라서 layer의 node num만 사용하고, 초기화는 1번부터 실행한다.
    for (int layer_idx = 1; layer_idx < LAYER_NUM; layer_idx++) {

        // weight에 대한 동적할당
        malloc_2dim_float_matrix(
            &(layers[layer_idx].weight), 
            layer_node_num[layer_idx], 
            layer_node_num[layer_idx-1]
        );

        // dW에 대한 동적할당
        malloc_2dim_float_matrix(
            &(layers[layer_idx].dW),
            layer_node_num[layer_idx],
            layer_node_num[layer_idx - 1]
        );

        // bias에 대한 동적할당
        malloc_1dim_float_vector(
            &(layers[layer_idx].bias),
            layer_node_num[layer_idx]
        );

        // db에 대한 동적할당
        malloc_1dim_float_vector(
            &(layers[layer_idx].db),
            layer_node_num[layer_idx]
        );
    }
    // ----- weight/dw, bias/db에 대한 동적할당 완료 -----

    // random 함수에 대한 seed값 설정
    srand(1);

    // 동적할당이 완료되었으므로, 실제 값을 채워넣는다.
    // hidden layer는 ReLU를 사용하므로, He Initialization을 사용한다.
    // output layer는 Softmax를 사용하므로, Xavier Initialization을 사용한다.
    for (int layer_idx = 1; layer_idx < LAYER_NUM; layer_idx++) {
        float std; // 표준편차
        float random_number; // random 변수

        // 1. weight/dW initialization
        // output layer의 weight/dW 초기화 (Xavier)
        // 단, 실제로는 Xavier가 아닌 LeCun 방식이지만, 참조한 교재에서 Xavier 방식이라고 적혀있어
        // 본 프로젝트에서는 LeCun을 Xavier라고 사용한다.
        if (layer_idx == LAYER_NUM - 1) {
            std = sqrt(1.0 / layer_node_num[layer_idx - 1]); // Xavier 표준편차

            for (int i = 0; i < layer_node_num[layer_idx]; i++) { // 현재 layer의 길이만큼
                for (int j = 0; j < layer_node_num[layer_idx - 1]; j++) { // 이전 layer의 길이만큼
                    random_number = get_random_number(std);
                    layers[layer_idx].weight[i][j] = random_number; // weight 추가
                    layers[layer_idx].dW[i][j] = 0; // dW는 gradient를 보관하므로, 0으로 초기화
                }
            }
        }
        // hidden layer의 weight/dW 초기화 (He)
        else {
            std = sqrt(2.0 / layer_node_num[layer_idx - 1]); // He 표준편차

            for (int i = 0; i < layer_node_num[layer_idx]; i++) { // 현재 layer의 길이만큼
                for (int j = 0; j < layer_node_num[layer_idx - 1]; j++) { // 이전 layer의 길이만큼
                    random_number = get_random_number(std);
                    layers[layer_idx].weight[i][j] = random_number; // weight 추가
                    layers[layer_idx].dW[i][j] = 0; // dW는 gradient를 보관하므로, 0으로 초기화
                }
            }
        }

        // 2. bias/db initialization
        // bias는 hidden/output에 관계없이 모두 0으로 초기화한다.
        for (int i = 0; i < layer_node_num[layer_idx]; i++) { // 현재 layer의 길이만큼
            layers[layer_idx].bias[i] = 0;
            layers[layer_idx].db[i] = 0;
        }
    }
}

void free_allocated_memory(Layer* layers, int* layer_node_num) {
    /*
        학습 과정에서 사용한 모든 동적할당 메모리를 해제하는 함수
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
        ReLU를 현재 들어온 값에 대해 element wise로 적용하는 함수
    */

    for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
        activation[node_index] = relu(activation[node_index]);
    }
}

float max(float* activation) {
    /*
        전달받은 float vector(output)에서 가장 큰 값을 찾아 반환하는 함수
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
        전달받은 float vector(output)에서 가장 큰 값의 index를 반환하는 함수
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
    
    float total_exp_value_sum = 0; // activation을 구성하는 모든 value에 exp를 취한 값의 합
    float max_value = max(activation); // activation에서 가장 큰 값을 구한다.

    for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
        // activation에서 max_value를 뺀 것에 exp를 취한 값을 구한다.
        double value = activation[node_index] - max_value;
        float exp_value = exp(value);

        // 해당 값을 total_exp_value_sum에 더한다.
        total_exp_value_sum += exp_value;

        // activation[node_index]를 exp_value로 바꾼다.
        // 다음 for loop에서 total_exp_value_sum으로 나눠줄 것이다.
        activation[node_index] = exp_value;
    }

    // activation의 각 value를 total_exp_value_sum으로 나눠준다.
    for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
        activation[0] /= total_exp_value_sum;
    }
}

void dot_product(float* activation, int prev_layer_node_num, int current_layer_node_num, float** weight) {
    // activation을 보관할 임시 vector 생성
    float* output = (float*)malloc(sizeof(float) * current_layer_node_num);

    // dot product 진행
    for (int i = 0; i < current_layer_node_num; i++) {
        float value = 0;

        for (int j = 0; j < prev_layer_node_num; j++) {
            value += weight[i][j] * activation[j];
        }

        output[i] = value;
    }

    // output의 값을 activation에 복사한다.
    copy_memory_float_1dim(activation, output, current_layer_node_num);

    // 동적할당 메모리 해제
    free(output);
}

void forward_pass(float* activation, int* layer_node_num, float (*data)[DATA_SIZE], int data_index, Layer* layers) {
    // 현재 data의 값을 activation으로 복사한다. (초기 input값)
    
    // data에서 activation으로 값을 복사한다.
    for (int i = 0; i < DATA_SIZE; i++) {
        activation[i] = data[data_index][i];
    }
    
    // for loop에서 이전 레이어의 출력을 저장할 변수
    float prev_activation[DATA_SIZE];

    for (int layer_index = 1; layer_index < LAYER_NUM; layer_index++) {
        // prev_activation에 activation의 값을 복사하여 저장한다.
        // loop를 반복하며, DATA_SIZE에서 점차 사용되는 공간이 줄어드는 형식.
        copy_memory_float_1dim(prev_activation, activation, layer_node_num[layer_index - 1]);

        int prev_layer_node_num = layer_node_num[layer_index-1];
        int current_layer_node_num = layer_node_num[layer_index];
        
        // dot product를 진행한다. 함수 실행이 끝나면, 
        // 변수 activation에는 새로운 layer의 Z값이 들어있다.
        dot_product(activation, prev_layer_node_num, current_layer_node_num, layers[layer_index].weight);
        
        // Z에 대해 bias를 각각 더해준다.
        for (int node_index = 0; node_index < current_layer_node_num; node_index++) {
            activation[node_index] += layers[layer_index].bias[node_index];
        }

        // 현재까지는 activation function을 거치지 않았으므로,
        // activation 변수가 Z값을 가지고 있는 상황이다.
        // 따라서 activation의 값을 cache_Z에 저장한다.
        copy_memory_float_1dim(layers[layer_index].cache_Z, activation, current_layer_node_num);

        // 또한 prev_A의 값을 cache에 저장한다.
        copy_memory_float_1dim(layers[layer_index].cache_prev_A, prev_activation, prev_layer_node_num);

        // activation function 적용
        // 현재 Layer의 activation function을 판단하여 activation을 적용한다.
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
        Softmax를 Activation Function으로 사용하는
        Output Layer에 대해 dZ를 구한다.
        (dZ는 Chain Rule 과정에서 Loss를 Z로 미분한 것을 의미한다.)
    */

    for (int node_idx = 0; node_idx < TARGET_ALPHABET_NUM; node_idx++) {
        dZ[node_idx] = activation[node_idx] - current_target[node_idx];
    }
}

void calc_hidden_layer_dZ(float* dZ, float* dA, int current_layer_idx, int current_layer_node_num, Layer* layers) {
    /*
        ReLU를 Activation Function으로 사용하는
        Output Layer에 대해 dZ를 구한다.
        (dZ는 Chain Rule 과정에서의 Loss를 Z로 미분한 것을 의미한다.)
    */

    // cache로부터 사전에 저장해 둔 Z값을 가져온다.
    float* Z = (float*)malloc(sizeof(float)*current_layer_node_num);
    copy_memory_float_1dim(Z, layers[current_layer_idx].cache_Z, current_layer_node_num);
    
    // Z의 size는 (curent_layer_node_num, 1)이다. 따라서 다음과 같이 for loop을 돌고,
    // ReLU의 gradient는 값이 양수라면 1, 0또는 음수라면 0이므로 아래와 같이 Z의 값을 변환한다.
    for (int node_idx = 0; node_idx < current_layer_node_num; node_idx++) {
        if (Z[node_idx] < 0) {
            Z[node_idx] = 0;
        }
        else {
            Z[node_idx] = 1;
        }
    }

    // 위에서 구한 ReLU의 미분값을 dA와 곱해서 dZ를 구한다. 
    for (int node_idx = 0; node_idx < current_layer_node_num; node_idx++) {
        dZ[node_idx] = dA[node_idx] * Z[node_idx];
    }

    // cache의 값을 가져오기 위해 사용했던 Z의 메모리를 해제한다.
    free(Z);
}

void calc_dW(float* dZ, int current_layer_idx, int current_layer_node_num, int prev_layer_node_num, Layer* layers) {
    /*
        인자로 전달받은 값을 사용해 layer의 dW를 업데이트 하는 함수
    */

    // 수식을 통해서 dW값 업데이트 진행
    for (int i = 0; i < current_layer_node_num; i++) {
        for (int j = 0; j < prev_layer_node_num; j++) {
            float dW_value = dZ[i] * layers[current_layer_idx].cache_prev_A[j];
            layers[current_layer_idx].dW[i][j] += dW_value;
        }
    }
}
void calc_db(float* dZ, int current_layer_idx, int current_layer_node_num, Layer* layers) {
    /*
        인자로 전달받은 값을 사용해 layer의 db를 업데이트 하는 함수
    */

    // db는 dZ의 값과 똑같으므로, 그대로 더한다.
    for (int i = 0; i < current_layer_node_num; i++) {
        layers[current_layer_idx].db[i] += dZ[i];
    }
}

void calc_dA_prev(float* dA_prev, float* dZ, int current_layer_idx, 
    int current_layer_node_num, int prev_layer_node_num, Layer* layers) {
    /*
        이전 레이어로 전달할 dA_prev의 값을 계산하는 함수
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
        softmax layer는 output layer이므로, forward pass에서의 최종 activation(output)을 전달받아
        gradient를 계산해 layer의 gradient 값을 업데이트한다.
    */

    // 현재 train_data에 대한 target 값을 복사하여 가져온다.
    int current_target[TARGET_ALPHABET_NUM];
    copy_memory_int_1dim(current_target, train_target[data_index], TARGET_ALPHABET_NUM);

    // Softmax는 Output layer에 사용된다고 생각하기 때문에,
    // layer의 index를 LAYER_NUM-1로 설정하며,
    // node의 개수도 TARGET_ALPHABET_NUM으로 설정한다.
    int current_layer_idx = LAYER_NUM - 1;
    int current_layer_node_num = TARGET_ALPHABET_NUM;
    
    int prev_layer_idx = current_layer_idx - 1;
    int prev_layer_node_num = layer_node_num[prev_layer_idx];

    // output layer이므로, dZ의 크기는 TARGET_ALPHABET_NUM이다.
    float dZ[TARGET_ALPHABET_NUM];

    // dW와 db를 계산한다.
    // 1) dW와 db를 계산하기 위해서는 dZ가 필요하므로, 먼저 dZ를 계산한다.
    // 2) dZ와 prev_activation을 바탕으로 dW를 계산한다.
    // 3) dZ를 바탕으로 db를 계산한다.
    calc_output_layer_dZ(dZ, current_target, activation); // dZ vector에 값이 저장된 상태. (index 0~6)
    calc_dW(dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers); // dW를 계산하고 업데이트
    calc_db(dZ, current_layer_idx, current_layer_node_num, layers); // db를 계산하고 업데이트
    
    // dZ와 Weight를 이용해 이전 Layer로 전달해 줄 dA_prev(prev_activation의 gradient)를 계산한다.
    calc_dA_prev(dA_prev, dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers);
}

void calc_relu_gradient(float* dA, int* layer_node_num, int current_layer_idx, Layer* layers) {
    /*
        ReLU에 대한 Gradient를 계산하는 함수
        현재 모델에서는 Hidden Layer가 Activation Function으로 ReLU를 사용하므로,
        특정 Hidden Layer에 대한 Gradient를 계산하는 것으로 생각할 수 있다.
    */

    int current_layer_node_num = layer_node_num[current_layer_idx];

    int prev_layer_idx= current_layer_idx - 1;
    int prev_layer_node_num = layer_node_num[prev_layer_idx];

    // dZ값을 저장하기 위한 vector를 동적할당한다.
    float* dZ = (float*)malloc(sizeof(float) * current_layer_node_num);

    // dW와 db를 계산한다.
    // 1) dW와 db를 계산하기 위해서는 dZ가 필요하므로, 먼저 dZ를 계산한다.
    // 2) dZ와 prev_activation을 바탕으로 dW를 계산한다.
    // 3) dZ를 바탕으로 db를 계산한다.
    calc_hidden_layer_dZ(dZ, dA, current_layer_idx, current_layer_node_num, layers); // dZ vector에 값이 저장된 상태
    calc_dW(dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers); // dW를 계산하고 업데이트
    calc_db(dZ, current_layer_idx, current_layer_node_num, layers); // db를 계산하고 업데이트
    calc_dA_prev(dA, dZ, current_layer_idx, current_layer_node_num, prev_layer_node_num, layers); // 이전 레이어로 보낼 dA_prev 계산

    // 임시로 활용한 메모리 해제
    free(dZ);
}

void backward_pass(float* activation, int* layer_node_num, int(*train_target)[TARGET_ALPHABET_NUM], int data_index, Layer* layers) {
    /*
        backward pass의 전체 과정을 실행하는 함수
        한 개의 example에 대한 backward pass를 수행한다.
    */

    // activation은 output 값을 backward pass로 넘겨주는 역할이다.
    // 따라서 값을 온전하게 넘겨줬다면, 그 역할이 끝났다고 볼 수 있다.

    // backward_pass에서 최초에 current_layer_idx는 output_layer_idx와 같다.
    int current_layer_idx = LAYER_NUM - 1;
    int current_layer_node_num = layer_node_num[current_layer_idx];
    
    int prev_layer_idx = current_layer_idx - 1;
    int prev_layer_node_num = layer_node_num[prev_layer_idx];

    // gardient를 구하는 과정에서, dA_prev를 계산하여
    // 데이터를 저장하는 역할을 하는 vector.
    // dA_prev의 최대 크기는 DATA_SIZE이므로 size를 이와 같이 지정한다.
    float dA_prev[DATA_SIZE];

    // calc_softmax_gradient를 거치면 output layer에 대한 dW/db가 저장되고,
    // dA_prev에 이전 hidden layer로 넘겨줄 값이 저장된다.
    calc_softmax_gradient(dA_prev, activation, layer_node_num, train_target, data_index, layers);

    for (int layer_idx = (LAYER_NUM - 1) - 1; layer_idx > 0; layer_idx--) {
        // hidden layer에 대한 gradient를 계산한다.
        current_layer_idx = layer_idx;

        // 현재 layer에 대해서 gradient가 구해지고, dA_prev에 값이 저장된다.
        // 저장된 dA_prev 값은 다음번 loop에서 prev layer의 activation gradient로 사용된다.
        calc_relu_gradient(dA_prev, layer_node_num, current_layer_idx, layers);
    }
}

void update_parameter(Layer* layers, int* layer_node_num) {
    /*
        저장된 gradient값을 바탕으로 parameter를 업데이트하고,
        값을 모두 0으로 초기화하는 함수
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
        forward pass의 결과가 정답인지 확인하는 함수
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
        Model training의 전체 과정을 수행하는 함수
    */

    // Accuracy 계산을 위해 float type의 train data set의 개수 변수 선언
    float total_train_data_num = 420.0;

    // train 진행, 정해진 epoch 수만큼 for loop을 실행한다.
    for (int epoch = 0; epoch < EPOCH_NUM; epoch++) {
        printf("epoch : %d\n", epoch);
        
        // Accuracy 계산을 위해 정답 개수를 세는 카운터 변수 선언
        float correct = 0.0;

        // 보유한 train data 개수만큼 for loop 실행. 한 번의 epoch을 의미한다.
        for (int train_data_index = 0; train_data_index < TRAIN_DATA_NUM; train_data_index++) {

            // forward pass의 output을 받아올 activation vector 생성
            // Layer의 노드 최대 크기가 DATA_SIZE이므로, 
            // activation vector의 크기를 DATA_SIZE로 만든다.
            // 이후에 backward_pass에 사용되며 Layer의 노드 개수가 증가될 때마다 
            // indexing을 적절히하여 값을 사용한다.
            float activation[DATA_SIZE];

            // 한 개의 example에 대하여 forward pass 수행
            forward_pass(activation, layer_node_num, train_data, train_data_index, layers);

            // forward_pass의 결과가 정답인지 확인
            if (isCorrect(activation, layer_node_num, train_target, train_data_index)) {
                correct += 1;
            }

            // 한 개의 example에 대하여 backward pass 수행
            backward_pass(activation, layer_node_num, train_target, train_data_index, layers);
        }

        // 한 번의 epoch에서 얻은 결과로 Accuracy를 계산
        printf("Accuracy : %f (%c)\n", (correct/total_train_data_num)*100.0, '%');
        
        // backward pass를 거쳐 구한 gradient를 바탕으로, 1 epoch에 대한 parameter update를 진행
        update_parameter(layers, layer_node_num);
    }
}

void predict(int* layer_node_num, float(*test_data)[DATA_SIZE], int(*test_target)[TARGET_ALPHABET_NUM], Layer* layers) {
    /*
        주어진 test data set에 대해서 결과를 예측하는 함수 (Test)
    */

    // Accuracy 계산을 위해 float type의 test data set의 개수 변수 선언
    float total_test_data_num = 140.0;

    // test 진행
    float correct = 0.0;

    for (int test_data_index = 0; test_data_index < TEST_DATA_NUM; test_data_index++) {
        // forward pass의 output을 받아올 activation vector 생성
        // Layer의 노드 최대 크기가 DATA_SIZE이므로, 
        // activation vector의 크기를 DATA_SIZE로 만든다.
        // 이후에 backward_pass에 사용되며 Layer의 노드 개수가 증가될 때마다 
        // indexing을 적절히하여 값을 사용한다.
        float activation[DATA_SIZE];

        forward_pass(activation, layer_node_num, test_data, test_data_index, layers);

        if (isCorrect(activation, layer_node_num, test_target, test_data_index)) {
            correct += 1;
        }

        backward_pass(activation, layer_node_num, test_target, test_data_index, layers);
    }

    // 결과 출력
    printf("\n\n\nTest Accuracy : %f (%c)\n", (correct / total_test_data_num) * 100.0, '%');
}