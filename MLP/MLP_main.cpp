#include "MLP.h"

int main()
{
	vector<vector<double>> feature = { {1,2,3},{4,5,6},{7,8,9},{2,2,3},{2,3,4},{5,1,2},{5,1,17},{1,2,7},{8,6,9},{2,2,7},{ 3,2,7 },{4,2,7},{5,3,7},{8,10,2},{0,1,5},{0,2,4},{11,0,4} };
	vector<vector<double>> result = { {6},{27},{66},{9},{11},{28},{43},{10},{81},{13},{18},{25},{35},{76},{6},{6},{125} };
	MLP mlp(feature, result);
	mlp.ShowData();
	vector<UI>arr = { 6, 8, 10, 7 ,4 };
	mlp.Ini_MLP(arr);
	mlp.All_batch(1000000, 0.0005);
	mlp.Show_Predict_result();
	vector<double>f = { 3,3,9 };
	vector<double>r = mlp.Predict(f);
	cout << r[0] << endl;
}