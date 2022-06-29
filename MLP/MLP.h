#include <iostream>
#include <vector>
#include <iomanip>
#define UI unsigned int
using namespace std;


class MLP
{
private:
	//���ݼ�������
	UI data_num;

	/****************************************/
	//������
	vector<vector<double>>feature;
	//��������
	UI feature_num;
	/****************************************/

	/****************************************/
	//�����
	vector<vector<double>>result;
	//���������
	UI result_num;
	/****************************************/

	/****************************************/
	//���ز�ת��
	bool ini_mlp;

	vector<vector<vector<double>>>neural_network;//�������(ת������)
	//�˴�˵��һ��neural_network����ά�ȵ�����
	//neural_network��������,neural_network[i]�Ǵӵ�i-1��(��0���������)����i���ת������
	//neural_network[i][j]��ʾ��i-1��ת������i���j��Ԫ�ص�ת�����󣬳���Ϊ(��i-1����Ԫ����+1)
	//neural_network[i][j][k]��ʾת������i���j��ʱ����i-1���k��Ԫ�ص�Ȩ�أ����һ����ƫ��


	vector<vector<vector<double>>>data_network;//���ݼ���
	//�˴�˵��һ�����ݼ��ϸ���ά�ȵ�����
	//data_networkΪ�����ݼ�����0��Ϊ�������ݼ�
	//data_network[i]Ϊ��i���������ݶ�Ӧ�������ݼ�
	//data_network[i][j]��ʾ��i���������ݶ�Ӧ�ĵ�j����Ԫ������
	//data_network[i][j][k]��ʾ��i���������ݶ�Ӧ�ĵ�j���k����Ԫ������
	//ע�⣬���һ����Ԥ�������ݼ������ǽ�����ݼ�

	vector<UI>neuron_num;//ÿһ�����Ԫ����(��ӽ����������)

	vector<vector<vector<double>>>delta;//����ά���ݶȵ��м�ֵ

	UI nef_num;//��Ԫ����
	/****************************************/

	vector<vector<double>>Predict_result;//Ԥ����

public:

	//�вγ�ʼ��
	MLP(vector<vector<double>>f, vector<vector<double>>r);

	//��ʾ����
	void ShowData();

	//network��ʼ��
	void Ini_MLP(vector<UI>arr);

	//������չʾ
	void Show_MLP();

	//����������ʾ
	void Show_DataNetwork();

	//��ʧ����
	double Loss(UI index);

	//��ʾdelta����
	void Show_Delta();

	//��ʧ�������ݶȣ�����һ��һά����
	vector<double> Gradent_Loss(UI index);

	/********************************************/
	//�������ϣ�������Ҫһ���
	//�����
	double Activate(double ini);

	//������ݶ�
	double Gradent_Activate(double ini);
	/********************************************/

	/********************************************/
	//�������ϣ�������Ҫһ���
	//ĩ��ת������
	//��ͬ���������������Ԥ������ת���ǲ�һ����
	double Result_Change(double ini);

	//ĩ��ת�������ݶ�
	double Gradent_Result_Change(double ini);
	/********************************************/

	//ǰ�򴫵ݺ���
	void Push_Forward(UI i);

	//���򴫲�����
	void Push_Back(UI index);

	//����������(�õڼ������ݴ���������(ѵ������)������������
	void Update_Neural_Network(UI index, double step, UI max_iter);

	//��ȡdatanum
	UI Get_data_num();

	//��ȡԤ����
	void Show_Predict_result();

	//mimi_batch
	void Mini_batch(UI pick_time, UI max_iter, double step);
};