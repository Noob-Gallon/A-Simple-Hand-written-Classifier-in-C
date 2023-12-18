#pragma once

// 데이터를 불러오는데 사용할 Buffer의 Size
#define BUFFER_SIZE 1025

// 사용할 이미지 데이터의 Row/Col Size
#define IMAGE_ROW_SIZE 16
#define IMAGE_COL_SIZE 16

// Training/Test Data의 개수
#define TRAIN_DATA_NUM 420
#define TEST_DATA_NUM 140

// 사용할 Data Example 하나의 길이
#define DATA_SIZE 256

// Target Label을 구성할 알파벳의 개수.
// output의 길이로서 사용된다.
#define TARGET_ALPHABET_NUM 7

// Hyper-Parameter
#define EPOCH_NUM 130
#define LEARNING_RATE 0.02
#define LAYER_NUM 4 // input layer를 포함한 전체 layer의 길이

// Activation Function
#define RELU 0
#define SOFTMAX 1

// Layer에 필요한 값들을 저장하는 구조체
typedef struct layer {
	int layer_idx;
	int activation_type; // 0 or 1, MACRO

	float* cache_Z;
	float* cache_prev_A;

	float** weight;
	float** dW;

	float* bias;
	float* db;
} Layer;

// Model에 필요한 값들을 저장하는 구조체
typedef struct model {
	Layer layers[LAYER_NUM]; // Layer 구조체 객체의 list

	int layer_node_num[LAYER_NUM];

	float train_data[TRAIN_DATA_NUM][DATA_SIZE];
	float test_data[TEST_DATA_NUM][DATA_SIZE];

	int train_target[TRAIN_DATA_NUM][TARGET_ALPHABET_NUM];
	int test_target[TEST_DATA_NUM][TARGET_ALPHABET_NUM];
} Model;

float max(float* activation);
float relu(float value);
void element_wise_relu(float* activation, int current_layer_node_num);
void softmax(float* activation, int current_layer_node_num);
void dot_product(float* activation, int prev_node_num, int current_node_num, float** weight);

void malloc_2dim_float_matrix(float*** matrix, int row_size, int col_size);
void malloc_1dim_float_vector(float** vector, int vector_size);

void copy_memory_int_1dim(int* dest, int* source, int len);
void copy_memory_float_1dim(float* dest, float* source, int len);

void initialize_layer_node_num(int* arr, int* value);
void initialize_layer(Layer* layers, int* layer_node_num);
float get_random_number(float std);
void initialize_parameter(Layer* layers, int* layer_node_num);
void free_allocated_memory(Layer* layers, int* layer_node_num);

void load_data_from_csv(float (*data)[DATA_SIZE], int (*target)[TARGET_ALPHABET_NUM], char* path, int data_index);
void get_file_name_by_index(char* file_name, int data_index, int path_name_len);
void load_train_data(float (*train_data)[DATA_SIZE], int (*train_target)[TARGET_ALPHABET_NUM]);
void load_test_data(float (*test_data)[DATA_SIZE], int(*test_target)[TARGET_ALPHABET_NUM]);
void load_target(int (*target)[TARGET_ALPHABET_NUM], int data_index, char label);
void load_data(
	float(*train_data)[DATA_SIZE], int(*train_target)[TARGET_ALPHABET_NUM], 
	float(*test_data)[DATA_SIZE], int(*test_target)[TARGET_ALPHABET_NUM]
);

void calc_output_layer_dZ(float* dZ, int* current_target, float* activation);
void calc_hidden_layer_dZ(float* dZ, float* dA, int current_layer_idx, int current_layer_node_num, Layer* layers);
void calc_dW(float* dZ, int current_layer_idx, int current_layer_node_num, int prev_layer_node_num, Layer* layers);
void calc_db(float* dZ, int current_layer_idx, int current_layer_node_num, Layer* layers);
void calc_dA_prev(float* dA_prev, float* dZ, int current_layer_idx, int current_layer_node_num, int prev_layer_node_num, Layer* layers);
void calc_softmax_gradient(float* dA_prev, float* activation, int* layer_node_num, 
	int(*train_target)[TARGET_ALPHABET_NUM], int data_index, Layer* layers);
void calc_relu_gradient(float* dA_prev, int* layer_node_num, int current_layer_idx, Layer* layers);
void update_parameter(Layer* layers, int* layer_node_num);
bool isCorrect(float* activation, int* layer_node_num, int(*train_target)[TARGET_ALPHABET_NUM], int train_data_index);
void forward_pass(float* activation, int* layer_node_num, float (*data)[DATA_SIZE], int data_index, Layer* layers);
void backward_pass(float* activation, int* layer_node_num, int (*train_target)[TARGET_ALPHABET_NUM], int data_index, Layer* layers);
void train(int* layer_node_num, float (*train_data)[DATA_SIZE], int (*train_target)[TARGET_ALPHABET_NUM], Layer* layers);
void predict(int* layer_node_num, float (*test_data)[DATA_SIZE], int(*test_target) [TARGET_ALPHABET_NUM], Layer* layers);