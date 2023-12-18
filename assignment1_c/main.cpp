#include "model_function.h" // 학습 과정에서 필요한 모든 함수를 정의한 헤더 파일

int main(void) {
	// 모델에 대한 정보를 가진 Model 구조체 변수 선언
	Model model;

	// 데이터 로드
	load_data(
		model.train_data, 
		model.train_target, 
		model.test_data, 
		model.test_target
	);

	// 학습에 사용할 레이어의 노드 개수
	// input layer를 포함한 개수이다.
	int layer_node_num[LAYER_NUM] = { 256, 96, 48, 7 };

	// layer 별로 node 개수를 초기화
	initialize_layer_node_num(model.layer_node_num, layer_node_num);

	// layer 별로 기본 설정값들을 초기화하고 배열을 동적할당한다.
	// 그리고 parameter를 random 초기화한다.
	initialize_layer(model.layers, model.layer_node_num);
	initialize_parameter(model.layers, model.layer_node_num);

	// train 시작
	train(model.layer_node_num, model.train_data, model.train_target, model.layers);

	 // test 시작
	 predict(model.layer_node_num, model.test_data, model.test_target, model.layers);

	// 학습에 사용된 전체 동적할당 메모리 해제
	free_allocated_memory(model.layers, model.layer_node_num);

	return 0;
}