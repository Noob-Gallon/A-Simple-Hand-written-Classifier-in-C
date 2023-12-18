#include "model_function.h" // �н� �������� �ʿ��� ��� �Լ��� ������ ��� ����

int main(void) {
	// �𵨿� ���� ������ ���� Model ����ü ���� ����
	Model model;

	// ������ �ε�
	load_data(
		model.train_data, 
		model.train_target, 
		model.test_data, 
		model.test_target
	);

	// �н��� ����� ���̾��� ��� ����
	// input layer�� ������ �����̴�.
	int layer_node_num[LAYER_NUM] = { 256, 96, 48, 7 };

	// layer ���� node ������ �ʱ�ȭ
	initialize_layer_node_num(model.layer_node_num, layer_node_num);

	// layer ���� �⺻ ���������� �ʱ�ȭ�ϰ� �迭�� �����Ҵ��Ѵ�.
	// �׸��� parameter�� random �ʱ�ȭ�Ѵ�.
	initialize_layer(model.layers, model.layer_node_num);
	initialize_parameter(model.layers, model.layer_node_num);

	// train ����
	train(model.layer_node_num, model.train_data, model.train_target, model.layers);

	 // test ����
	 predict(model.layer_node_num, model.test_data, model.test_target, model.layers);

	// �н��� ���� ��ü �����Ҵ� �޸� ����
	free_allocated_memory(model.layers, model.layer_node_num);

	return 0;
}