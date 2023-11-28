#ifndef KAERMAN_H_
#define KAERMAN_H_
#include <iostream>
using namespace std;
// 一维滤波器信息结构体
typedef  struct{
	double filterValue;  //k-1时刻的滤波值，即是k-1时刻的值
	double kalmanGain;   //   Kalamn增益
	double A;   // x(n)=A*x(n-1)+u(n),u(n)~N(0,Q)
	double H;   // z(n)=H*x(n)+w(n),w(n)~N(0,R)
	double Q;   //预测过程噪声偏差的方差
	double R;   //测量噪声偏差，(系统搭建好以后，通过测量统计实验获得)
	double P;   //估计误差协方差
}  KalmanInfo;

void Init_KalmanInfo(KalmanInfo* info, double Q, double R)
{
	info->A = 1;  //标量卡尔曼
	info->H = 1;  //
	info->P = 10;  //后验状态估计值误差的方差的初始值（不要为0问题不大）
	info->Q = Q;    //预测（过程）噪声方差 影响收敛速率，可以根据实际需求给出
	info->R = R;    //测量（观测）噪声方差 可以通过实验手段获得
	info->filterValue = 0;// 测量的初始值
}
double KalmanFilter(KalmanInfo* kalmanInfo, double lastMeasurement)
{
	//预测下一时刻的值
	double predictValue = kalmanInfo->A* kalmanInfo->filterValue;
	//x的先验估计由上一个时间点的后验估计值和输入信息给出，此处需要根据基站高度做一个修改

	//求协方差
	kalmanInfo->P = kalmanInfo->A*kalmanInfo->A*kalmanInfo->P + kalmanInfo->Q;
	//计算先验均方差 p(n|n-1)=A^2*p(n-1|n-1)+q

	double preValue = kalmanInfo->filterValue;  //记录上次实际坐标的值

	//计算kalman增益
	kalmanInfo->kalmanGain = kalmanInfo->P*kalmanInfo->H / (kalmanInfo->P*kalmanInfo->H*kalmanInfo->H + kalmanInfo->R);
	//Kg(k)= P(k|k-1) H’ / (H P(k|k-1) H’ + R)

	//修正结果，即计算滤波值
	kalmanInfo->filterValue = predictValue + (lastMeasurement - predictValue)*kalmanInfo->kalmanGain;  //利用残余的信息改善对x(t)的估计，给出后验估计，这个值也就是输出  X(k|k)= X(k|k-1)+Kg(k) (Z(k)-H X(k|k-1))

	//更新后验估计
	kalmanInfo->P = (1 - kalmanInfo->kalmanGain*kalmanInfo->H)*kalmanInfo->P;
	//计算后验均方差  P[n|n]=(1-K[n]*H)*P[n|n-1]

	return  kalmanInfo->filterValue;
}
#endif // KAERMAN_H_

