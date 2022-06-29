#include "MLP.h"

//�вγ�ʼ��
MLP::MLP(vector<vector<double>>f, vector<vector<double>>r)
{
	ini_mlp = false;
	feature = f;
	result = r;
	feature_num = feature[0].size();
	result_num = result[0].size();
	data_num = result.size();
	//���ݹ�һ������
	for (UI i = 0; i < feature_num; i++)
	{
		double Max = feature[0][i];
		double Min = feature[0][i];
		//Ѱ�������Сֵ
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

//����չʾ
void MLP::ShowData()
{
	cout << "������ size : [ " << feature_num << " x " << data_num << " ]" << endl;
	for (UI j = 0; j < data_num; j++)
	{
		for (UI i = 0; i < feature_num; i++)
		{
			cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << feature[j][i];
		}
		cout << endl;
	}
	cout << endl;
	cout << "����� size : [ " << result_num << " x " << data_num << " ]" << endl;
	for (UI j = 0; j < data_num; j++)
	{
		for (UI i = 0; i < result_num; i++)
		{
			cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << result[j][i];
		}
		cout << endl;
	}
}

//MLP��ʼ��
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
	nef_num = neuron_num.size();//Ӧ���������+�����+���ز��������
	//ת�������ʼ��
	for (UI i = 0; i < nef_num - 1; i++)
	{
		//ÿһ��ľ�����(ǰһ���ά��+1)x��һ���ά��
		vector<vector<double>>network;
		for (UI w = 0; w < neuron_num[i + 1]; w++)
		{
			vector<double>front;//һά��ǰ��ת������
			for (UI j = 0; j <= neuron_num[i]; j++)
			{
				front.push_back(double(rand() % 10 + 1) / 10);
			}
			network.push_back(front);
		}
		neural_network.push_back(network);
	}
	//�����ʼ�����
	//��ʼ�����ݴ洢��
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
	//���ݼ���ʼ�����
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

//��ʾMLP
void MLP::Show_MLP()
{
	if (!ini_mlp)
	{
		cout << "���ȳ�ʼ��ת���㺯��" << endl;
		return;
	}
	cout << "������ : " << endl;
	for (UI i = 0; i < nef_num - 1; i++)
	{
		cout << "��" << i << "->" << i + 1 << "��ת�Ļ����� : " << endl;
		for (UI w = 0; w < neuron_num[i + 1]; w++)
		{
			cout << "���" << i + 1 << "��" << "��" << w + 1 << "����Ԫ��ת������ : ";
			for (UI j = 0; j <= neuron_num[i]; j++)
			{
				cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << neural_network[i][w][j];
			}
			cout << endl;
		}
		cout << endl;
	}
}

//��ʾ��������
void MLP::Show_DataNetwork()
{
	if (!ini_mlp)
	{
		cout << "���ȳ�ʼ��ת���㺯��" << endl;
		return;
	}
	cout << "�������� : " << endl;
	for (UI i = 0; i < data_num; i++)
	{
		cout << "��" << i + 1 << "������ : " << endl;
		for (UI j = 0; j < nef_num; j++)
		{
			cout << "��" << j << "������";
			for (UI k = 0; k < neuron_num[j]; k++)
			{
				cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << data_network[i][j][k];
			}
			cout << endl;
		}
	}
	cout << endl;
}

//�����
double MLP::Activate(double ini)
{
	//ʹ��sigmoid��������
	double result = 1 / (1 + exp(-ini));
	return result;
}

//������ݶ�
double MLP::Gradent_Activate(double ini)
{
	double result = ini * (1 - ini);
	return result;
}

//ĩ��ת������
double MLP::Result_Change(double ini)
{
	return ini;
}

//ĩ��ת�������ݶ�
double MLP::Gradent_Result_Change(double ini)
{
	return 1;
}

//ǰ��ת������
void MLP::Push_Forward(UI i)
{
	//�Ȱ�ǰ��ȫ��ת��
		//��ÿһ�����ݼ����д���
	for (UI j = 0; j < nef_num - 2; j++)
	{
		//��㴫��
		for (UI k = 0; k < neuron_num[j + 1]; k++)
		{
			data_network[i][j + 1][k] = 0;
			//����ת��
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

//���򴫲�����(����delta)
void MLP::Push_Back(UI index)
{
	//�Ȱ����һ�������
	Gradent_Loss(index);
	for (int i = nef_num - 2; i >= 1; i--)
	{
		vector<double>temp;
		for (UI j = 0; j < neuron_num[i]; j++)
		{
			double r = 0;
			for (UI k = 0; k < neuron_num[i + 1]; k++)
			{
				r += neural_network[i][k][j] * delta[index][i + 1][k];//����i��i+1��ת������
			}
			temp.push_back(r * Gradent_Activate(data_network[index][i][j]));
		}
		delta[index][i] = temp;
	}
}

//����������
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

//��ʾdelta����
void MLP::Show_Delta()
{
	if (!ini_mlp)
	{
		cout << "���ȳ�ʼ��ת���㺯��" << endl;
		return;
	}
	cout << "delta : " << endl;
	for (UI i = 0; i < data_num; i++)
	{
		cout << "��" << i + 1 << "������ : " << endl;
		for (UI j = 0; j < nef_num; j++)
		{
			cout << "��" << j << "������";
			for (UI k = 0; k < neuron_num[j]; k++)
			{
				cout << fixed << setprecision(2) << setiosflags(ios::left) << setw(8) << delta[i][j][k];
			}
			cout << endl;
		}
	}
	cout << endl;
}

//��ʧ����
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

//��ʧ������ƫ��
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

//��ȡdatanum
UI MLP::Get_data_num()
{
	return data_num;
}

//��ȡԤ����
void MLP::Show_Predict_result()
{
	for (UI i = 0; i < Predict_result.size(); i++)
	{
		cout << "��" << i + 1 << "�����ݵ�Ԥ���� : ";
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
	UI total = data_num;//����
	//�����ȡ����
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
		//�������
		for (vector<UI>::iterator it = arr.begin(); it != arr.end(); it++)
		{
			Update_Neural_Network(*it, step, max_iter);
		}
	}
}