#include "MLP.h"

//有参初始化
MLP::MLP(vector<vector<double>>f, vector<vector<double>>r)
{
	ini_mlp = false;
	feature = f;
	result = r;
	feature_num = feature[0].size();
	result_num = result[0].size();
	data_num = result.size();
	//数据归一化处理
	for (UI i = 0; i < feature_num; i++)
	{
		double Max = feature[0][i];
		double Min = feature[0][i];
		//寻找最大最小值
		for (UI j = 0; j < data_num; j++)
		{
			if (feature[j][i] > Max)
			{
				Max = feature[j][i];
			}
			if (feature[j][i] < Min)
			{
				Min = feature[j][i];
			}
		}
		for (UI j = 0; j < data_num; j++)
		{
			feature[j][i] = (feature[j][i] - Min) / (Max - Min);
		}
	}
}

//数据展示
void MLP::ShowData()
{
	cout << "特征集 size : [ " << feature_num << " x " << data_num << " ]" << endl;
	for (UI j = 0; j < data_num; j++)
	{
		for (UI i = 0; i < feature_num; i++)
		{
			cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << feature[j][i];
		}
		cout << endl;
	}
	cout << endl;
	cout << "结果集 size : [ " << result_num << " x " << data_num << " ]" << endl;
	for (UI j = 0; j < data_num; j++)
	{
		for (UI i = 0; i < result_num; i++)
		{
			cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << result[j][i];
		}
		cout << endl;
	}
}

//MLP初始化
void MLP::Ini_MLP(vector<UI>arr)
{
	ini_mlp = true;
	srand((UI)time(0));
	neuron_num.push_back(feature_num);
	for (vector<UI>::iterator it = arr.begin(); it != arr.end(); it++)
	{
		neuron_num.push_back(*it);
	}
	neuron_num.push_back(result_num);
	nef_num = neuron_num.size();//应该是输入层+输出层+隐藏层的总数量
	//转化矩阵初始化
	for (UI i = 0; i < nef_num - 1; i++)
	{
		//每一层的矩阵都是(前一层的维度+1)x后一层的维度
		vector<vector<double>>network;
		for (UI w = 0; w < neuron_num[i + 1]; w++)
		{
			vector<double>front;//一维的前向转化函数
			for (UI j = 0; j <= neuron_num[i]; j++)
			{
				front.push_back(double(rand() % 10 + 1) / 10);
			}
			network.push_back(front);
		}
		neural_network.push_back(network);
	}
	//矩阵初始化完成
	//初始化数据存储集
	for (UI i = 0; i < data_num; i++)
	{
		vector<vector<double>>data_map;
		data_map.push_back(feature[i]);
		for (UI j = 1; j < nef_num; j++)
		{
			vector<double>temp;
			for (UI k = 0; k < neuron_num[j]; k++)
			{
				temp.push_back(0);
			}
			data_map.push_back(temp);
		}
		data_network.push_back(data_map);
	}
	//数据集初始化完成
	for (UI i = 0; i < data_num; i++)
	{
		vector<vector<double>>data_map;
		for (UI j = 0; j < nef_num; j++)
		{
			vector<double>temp;
			for (UI k = 0; k < neuron_num[j]; k++)
			{
				temp.push_back(0);
			}
			data_map.push_back(temp);
		}
		delta.push_back(data_map);
	}
	for (UI i = 0; i < data_num; i++)
	{
		vector<double>temp;
		for (UI j = 0; j < result_num; j++)
		{
			temp.push_back(0);
		}
		Predict_result.push_back(temp);
	}
}

//显示MLP
void MLP::Show_MLP()
{
	if (!ini_mlp)
	{
		cout << "请先初始化转化层函数" << endl;
		return;
	}
	cout << "神经网络 : " << endl;
	for (UI i = 0; i < nef_num - 1; i++)
	{
		cout << "第" << i << "->" << i + 1 << "层转的化矩阵 : " << endl;
		for (UI w = 0; w < neuron_num[i + 1]; w++)
		{
			cout << "向第" << i + 1 << "层" << "第" << w + 1 << "个神经元的转化数组 : ";
			for (UI j = 0; j <= neuron_num[i]; j++)
			{
				cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << neural_network[i][w][j];
			}
			cout << endl;
		}
		cout << endl;
	}
}

//显示数据网络
void MLP::Show_DataNetwork()
{
	if (!ini_mlp)
	{
		cout << "请先初始化转化层函数" << endl;
		return;
	}
	cout << "数据网络 : " << endl;
	for (UI i = 0; i < data_num; i++)
	{
		cout << "第" << i + 1 << "组数据 : " << endl;
		for (UI j = 0; j < nef_num; j++)
		{
			cout << "第" << j << "层数据";
			for (UI k = 0; k < neuron_num[j]; k++)
			{
				cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << data_network[i][j][k];
			}
			cout << endl;
		}
	}
	cout << endl;
}

//激活函数
double MLP::Activate(double ini)
{
	//使用sigmoid函数激活
	double result = 1 / (1 + exp(-ini));
	return result;
}

//激活函数梯度
double MLP::Gradent_Activate(double ini)
{
	double result = ini * (1 - ini);
	return result;
}

//末端转化函数
double MLP::Result_Change(double ini)
{
	return ini;
}

//末端转化函数梯度
double MLP::Gradent_Result_Change(double ini)
{
	return 1;
}

//前向转化函数
void MLP::Push_Forward(UI i)
{
	//先把前端全部转化
		//对每一个数据集进行传递
	for (UI j = 0; j < nef_num - 2; j++)
	{
		//逐层传递
		for (UI k = 0; k < neuron_num[j + 1]; k++)
		{
			data_network[i][j + 1][k] = 0;
			//数列转化
			for (UI w = 0; w < neuron_num[j]; w++)
			{
				data_network[i][j + 1][k] += data_network[i][j][w] * neural_network[j][k][w];
			}
			data_network[i][j + 1][k] += neural_network[j][k][neuron_num[j]];
			data_network[i][j + 1][k] = Activate(data_network[i][j + 1][k]);
		}
	}
	vector<double>predict;
	for (UI j = 0; j < neuron_num[nef_num - 1]; j++)
	{
		data_network[i][nef_num - 1][j] = 0;
		for (UI k = 0; k < neuron_num[nef_num - 2]; k++)
		{
			data_network[i][nef_num - 1][j] += data_network[i][nef_num - 2][k] * neural_network[nef_num - 2][j][k];
		}
		data_network[i][nef_num - 1][j] += neural_network[nef_num - 2][j][neuron_num[nef_num - 2]];
		predict.push_back(Result_Change(data_network[i][nef_num - 1][j]));
	}
	Predict_result[i] = predict;
}

//反向传播函数(构造delta)
void MLP::Push_Back(UI index)
{
	//先把最后一层算出来
	Gradent_Loss(index);
	for (int i = nef_num - 2; i >= 1; i--)
	{
		vector<double>temp;
		for (UI j = 0; j < neuron_num[i]; j++)
		{
			double r = 0;
			for (UI k = 0; k < neuron_num[i + 1]; k++)
			{
				r += neural_network[i][k][j] * delta[index][i + 1][k];//这是i到i+1的转化矩阵
			}
			temp.push_back(r * Gradent_Activate(data_network[index][i][j]));
		}
		delta[index][i] = temp;
	}
}

//更新神经网络
void MLP::Update_Neural_Network(UI index,double step,UI max_iter)
{
	for (UI i = 0; i < max_iter; i++)
	{
		Push_Forward(index);
		Push_Back(index);
		for (UI j = 0; j < nef_num - 1; j++)
		{
			for (UI w = 0; w < neuron_num[j + 1]; w++)
			{
				for (UI k = 0; k < neuron_num[j]; k++)
				{
					neural_network[j][w][k] = neural_network[j][w][k] - step * delta[index][j + 1][w] * data_network[index][j][k];
				}
				neural_network[j][w][neuron_num[j]] = neural_network[j][w][neuron_num[j]] - step * delta[index][j + 1][w];
			}
		}
	}
}

//显示delta数组
void MLP::Show_Delta()
{
	if (!ini_mlp)
	{
		cout << "请先初始化转化层函数" << endl;
		return;
	}
	cout << "delta : " << endl;
	for (UI i = 0; i < data_num; i++)
	{
		cout << "第" << i + 1 << "组数据 : " << endl;
		for (UI j = 0; j < nef_num; j++)
		{
			cout << "第" << j << "层数据";
			for (UI k = 0; k < neuron_num[j]; k++)
			{
				cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << delta[i][j][k];
			}
			cout << endl;
		}
	}
	cout << endl;
}

//损失函数
double MLP::Loss(UI index)
{
	double loss = 0;
	for (UI i = 0; i < neuron_num[nef_num - 1]; i++)
	{
		loss += (Predict_result[index][i] - result[index][i]) * (Predict_result[index][i] - result[index][i]);
	}
	loss = loss / 2 / neuron_num[nef_num - 1];
	return loss;
}

//损失函数求偏导
vector<double> MLP::Gradent_Loss(UI index)
{
	vector<double>temp;
	for (UI i = 0; i < neuron_num[nef_num - 1]; i++)
	{
		double loss_grand = (Predict_result[index][i] - result[index][i]) / neuron_num[nef_num - 1];
		loss_grand = Gradent_Result_Change(data_network[index][nef_num - 1][i]) * loss_grand;
		temp.push_back(loss_grand);
	}
	delta[index][nef_num - 1] = temp;
	return temp;
}

//获取datanum
UI MLP::Get_data_num()
{
	return data_num;
}

//获取预测结果
void MLP::Show_Predict_result()
{
	for (UI i = 0; i < Predict_result.size(); i++)
	{
		cout << "第" << i + 1 << "组数据的预测结果 : ";
		for (UI j = 0; j < Predict_result[i].size(); j++)
		{
			cout << Predict_result[i][j] << "  ";
		}
		cout << endl;
	}
}

//mini-batch
void MLP::Mini_batch(UI pick_time, UI max_iter, double step)
{
	srand(UI(time(0)));
	UI total = data_num;//总数
	//构造抽取数列
	for (UI i = 0; i < pick_time; i++)
	{
		vector<UI>arr;
		while (arr.size() < total * 0.2)
		{
			UI R = rand() % data_num;
			bool check = true;
			for (vector<UI>::iterator it = arr.begin(); it != arr.end(); it++)
			{
				if (*it == R)
				{
					check = false;
					break;
				}
			}
			if (check)
			{
				arr.push_back(R);
			}
		}
		//构造完成
		for (vector<UI>::iterator it = arr.begin(); it != arr.end(); it++)
		{
			Update_Neural_Network(*it, step, max_iter);
		}
	}
}