#include <iostream>
#include <vector>
#include <iomanip>
#define UI unsigned int
using namespace std;


class MLP
{
private:
	//数据集总数量
	UI data_num;

	/****************************************/
	//特征集
	vector<vector<double>>feature;
	//特征数量
	UI feature_num;
	/****************************************/

	/****************************************/
	//结果集
	vector<vector<double>>result;
	//结果集数量
	UI result_num;
	/****************************************/

	/****************************************/
	//隐藏层转化
	bool ini_mlp;

	vector<vector<vector<double>>>neural_network;//神经网络层(转化矩阵)
	//此处说明一下neural_network各个维度的意义
	//neural_network是总网络,neural_network[i]是从第i-1层(第0层是输入层)到第i层的转化矩阵
	//neural_network[i][j]表示第i-1层转化到第i层第j个元素的转化矩阵，长度为(第i-1层神经元数量+1)
	//neural_network[i][j][k]表示转化到第i层第j个时，第i-1层第k个元素的权重，最后一项是偏置


	vector<vector<vector<double>>>data_network;//数据集合
	//此处说明一下数据集合各个维度的意义
	//data_network为总数据集，第0层为特征数据集
	//data_network[i]为第i组输入数据对应的总数据集
	//data_network[i][j]表示第i组输入数据对应的第j层神经元的数据
	//data_network[i][j][k]表示第i组输入数据对应的第j层第k个神经元的数据
	//注意，最后一层是预测结果数据集，不是结果数据集

	vector<UI>neuron_num;//每一层的神经元数量(添加结果层和输入层)

	vector<vector<vector<double>>>delta;//用来维护梯度的中间值

	UI nef_num;//神经元层数
	/****************************************/

	vector<vector<double>>Predict_result;//预测结果

public:

	//有参初始化
	MLP(vector<vector<double>>f, vector<vector<double>>r);

	//显示数据
	void ShowData();

	//network初始化
	void Ini_MLP(vector<UI>arr);

	//神经网络展示
	void Show_MLP();

	//数据网络显示
	void Show_DataNetwork();

	//损失函数
	double Loss(UI index);

	//显示delta函数
	void Show_Delta();

	//损失函数求梯度，返回一个一维向量
	vector<double> Gradent_Loss(UI index);

	/********************************************/
	//激活函数组合，这两个要一起变
	//激活函数
	double Activate(double ini);

	//激活函数梯度
	double Gradent_Activate(double ini);
	/********************************************/

	/********************************************/
	//激活函数组合，这两个要一起变
	//末端转化函数
	//不同的问题对于最终神经预测结果的转化是不一样的
	double Result_Change(double ini);

	//末端转化函数梯度
	double Gradent_Result_Change(double ini);
	/********************************************/

	//前向传递函数
	void Push_Forward(UI i);

	//反向传播函数
	void Push_Back(UI index);

	//更新神经网络(用第几组数据处理，处理步长(训练精度)，最大迭代次数
	void Update_Neural_Network(UI index, double step, UI max_iter);

	//获取datanum
	UI Get_data_num();

	//获取预测结果
	void Show_Predict_result();

	//mimi_batch
	void Mini_batch(UI pick_time, UI max_iter, double step);
};