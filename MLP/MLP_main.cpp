#include "MLP.h"

int main()
{
	vector<vector<double>> feature = { {1,2,3},{4,5,6},{7,8,9},{10,11,12},{12,13,14} };
	vector<vector<double>> result = { {1,0,0},{0,1,0},{0,0,1},{0,1,0},{0,0,1} };
	MLP mlp(feature, result);
	mlp.ShowData();
	vector<UI>arr = { 4, 6, 5 ,5, 6 };
	mlp.Ini_MLP(arr);
	mlp.Mini_batch(100, 10000, 0.0001);
	mlp.Show_Predict_result();
}