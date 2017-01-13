#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include<string> 
#include <cmath>

// Namespace for using cout.
using namespace std;
// Namespace for using Mat
using namespace cv;

#define  innode 2  //输入结点数
#define  hidenode 10//隐含结点数
#define  outnode 1 //输出结点数
#define  trainsample 34//BP训练样本数

class BpNet
{
public:
	void train(double p[trainsample][innode ],double t[trainsample][outnode]);//Bp训练
	double p[trainsample][innode];     //输入的样本
	double t[trainsample][outnode];    //样本要输出的

	double *recognize(double *p);//Bp识别

	void writetrain(); //写训练完的权值
	void readtrain(); //读训练好的权值，这使的不用每次去训练了，只要把训练最好的权值存下来就OK

	BpNet();
	virtual ~BpNet();

public:
	void init();
	double w[innode][hidenode];//隐含结点权值
	double w1[hidenode][outnode];//输出结点权值
	double b1[hidenode];//隐含结点阀值
	double b2[outnode];//输出结点阀值

	double rate_w; //权值学习率（输入层-隐含层)
	double rate_w1;//权值学习率 (隐含层-输出层)
	double rate_b1;//隐含层阀值学习率
	double rate_b2;//输出层阀值学习率

	double e;//误差计算
	double error;//允许的最大误差
	double result[outnode];// Bp输出
};

BpNet::BpNet()
{
	error=1.0;
	e=0.0;

	rate_w=0.9;  //权值学习率（输入层--隐含层)
	rate_w1=0.9; //权值学习率 (隐含层--输出层)
	rate_b1=0.9; //隐含层阀值学习率
	rate_b2=0.9; //输出层阀值学习率
}

BpNet::~BpNet()
{

}

void winit(double w[],int n) //权值初始化（初始化所有的网络权值为-1 和 1 之间的随机数）
{
	for(int i=0;i<n;i++)
		w[i]=(2.0*(double)rand()/RAND_MAX)-1;
}

void BpNet::init()
{
	winit((double*)w,innode*hidenode);
	winit((double*)w1,hidenode*outnode);
	winit(b1,hidenode);
	winit(b2,outnode);
}

void BpNet::train(double p[trainsample][innode],double t[trainsample][outnode])
{
	double pp[hidenode];//隐含结点的校正误差
	double qq[outnode];//希望输出值与实际输出值的偏差
	double yd[outnode];//希望输出值

	double x[innode]; //输入向量
	double x1[hidenode];//隐含结点状态值
	double x2[outnode];//输出结点状态值
	double o1[hidenode];//隐含层激活值
	double o2[hidenode];//输出层激活值
	
	for(int isamp=0;isamp<trainsample;isamp++)//循环训练一次样本
	{
		for(int i=0;i<innode;i++)
			x[i]=p[isamp][i]; //输入的样本
		for(int i=0;i<outnode;i++)
			yd[i]=t[isamp][i]; //期望输出的样本

		//正向传播
			//计算隐含层输出
		for(int j=0;j<hidenode;j++)
		{
			o1[j]=0.0;
			for(int i=0;i<innode;i++)
				o1[j]=o1[j]+w[i][j]*x[i];//隐含层各单元输入激活值
			//激励函数
			x1[j]=1.0/(1+exp(-o1[j]-b1[j]));//隐含层各单元的输出

		}
			//计算输出层输出
		for(int k=0;k<outnode;k++)
		{
			o2[k]=0.0;
			for(int j=0;j<hidenode;j++)
				o2[k]=o2[k]+w1[j][k]*x1[j]; //输出层各单元输入激活值
			//激励函数
			x2[k]=1.0/(1.0+exp(-o2[k]-b2[k])); //输出层各单元的输出
			
		}
		
		//反向传播
		for(int k=0;k<outnode;k++)
		{
			//计算输出层误差
			qq[k]=(yd[k]-x2[k])*x2[k]*(1-x2[k]); //希望输出与实际输出的偏差
			//更新权值
			for(int j=0;j<hidenode;j++)
				w1[j][k]+=rate_w1*qq[k]*x1[j];  //下一次的隐含层和输出层之间的新连接权
		}

		for(int j=0;j<hidenode;j++)
		{
			//计算隐含层误差
			pp[j]=0.0;
			for(int k=0;k<outnode;k++)
				pp[j]=pp[j]+qq[k]*w1[j][k];
			pp[j]=pp[j]*x1[j]*(1-x1[j]); //隐含层的校正误差
			//更新权值
			for(int i=0;i<innode;i++)
				w[i][j]+=rate_w*pp[j]*x[i]; //下一次的输入层和隐含层之间的新连接权
		}

		for(int k=0;k<outnode;k++)
		{
			e+=fabs(yd[k]-x2[k])*fabs(yd[k]-x2[k]); //计算均方差(所有网络输出的误差求和)
		}
		error=e/2.0;
		//更新阈值
		for(int k=0;k<outnode;k++)
			b2[k]=b2[k]+rate_b2*qq[k]; //下一次的隐含层和输出层之间的新阈值
		for(int j=0;j<hidenode;j++)
			b1[j]=b1[j]+rate_b1*pp[j]; //下一次的输入层和隐含层之间的新阈值
	}
}

double *BpNet::recognize(double *p)
{
	double x[innode]; //输入向量
	double x1[hidenode]; //隐含结点状态值
	double x2[outnode]; //输出结点状态值
	double o1[hidenode]; //隐含层激活值
	double o2[hidenode]; //输出层激活值

	for(int i=0;i<innode;i++)
		x[i]=p[i];

	for(int j=0;j<hidenode;j++)
	{
		o1[j]=0.0;
		for(int i=0;i<innode;i++)
			o1[j]=o1[j]+w[i][j]*x[i]; //隐含层各单元激活值
		x1[j]=1.0/(1.0+exp(-o1[j]-b1[j])); //隐含层各单元输出
	}

	for(int k=0;k<outnode;k++)
	{
		o2[k]=0.0;
		for(int j=0;j<hidenode;j++)
			o2[k]=o2[k]+w1[j][k]*x1[j];//输出层各单元激活值
		x2[k]=1.0/(1.0+exp(-o2[k]-b2[k]));//输出层各单元输出
	}

	for(int k=0;k<outnode;k++)
	{
		result[k]=x2[k];
	}
	return result;
}

void BpNet::writetrain()
{
	FILE *stream0;
	FILE *stream1;
	FILE *stream2;
	FILE *stream3;
	int i,j;
	//隐含结点权值写入
	if(( stream0 = fopen("w.txt", "w+" ))==NULL)
	{
		cout<<"创建文件失败!";
		exit(1);
	}
	for(i=0;i<innode;i++)
	{
		for(j=0;j<hidenode;j++)
		{
			fprintf(stream0, "%f\n", w[i][j]);
		}
	}
	fclose(stream0);

	//输出结点权值写入
	if(( stream1 = fopen("w1.txt", "w+" ))==NULL)
	{
		cout<<"创建文件失败!";
		exit(1);
	}
	for(i=0;i<hidenode;i++)
	{
		for(j=0;j<outnode;j++)
		{
			fprintf(stream1, "%f\n",w1[i][j]);
		}
	}
	fclose(stream1);

	//隐含结点阀值写入
	if(( stream2 = fopen("b1.txt", "w+" ))==NULL)
	{
		cout<<"创建文件失败!";
		exit(1);
	}
	for(i=0;i<hidenode;i++)
		fprintf(stream2, "%f\n",b1[i]);
	fclose(stream2);

	//输出结点阀值写入
	if(( stream3 = fopen("b2.txt", "w+" ))==NULL)
	{
		cout<<"创建文件失败!";
		exit(1);
	}
	for(i=0;i<outnode;i++)
		fprintf(stream3, "%f\n",b2[i]);
	fclose(stream3);

}

void BpNet::readtrain()
{
	FILE *stream0;
	FILE *stream1;
	FILE *stream2;
	FILE *stream3;
	int i,j;

	//隐含结点权值读出
	if(( stream0 = fopen("w.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	float  wx[innode][hidenode];
	for(i=0;i<innode;i++)
	{
		for(j=0;j<hidenode;j++)
		{
			fscanf(stream0, "%f", &wx[i][j]);
			//cout<<"...........: "<<wx[i][j]<<endl<<endl;
			w[i][j]=wx[i][j];
		}
	}
	fclose(stream0);

	//输出结点权值读出
	if(( stream1 = fopen("w1.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	float  wx1[hidenode][outnode];
	for(i=0;i<hidenode;i++)
	{
		for(j=0;j<outnode;j++)
		{
			fscanf(stream1, "%f", &wx1[i][j]);
			w1[i][j]=wx1[i][j];
		}
	}
	fclose(stream1);

	//隐含结点阀值读出
	if(( stream2 = fopen("b1.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	float xb1[hidenode];
	for(i=0;i<hidenode;i++)
	{
		fscanf(stream2, "%f",&xb1[i]);
		b1[i]=xb1[i];
	}
	fclose(stream2);

	//输出结点阀值读出
	if(( stream3 = fopen("b2.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	float xb2[outnode];
	for(i=0;i<outnode;i++)
	{
		fscanf(stream3, "%f",&xb2[i]);
		b2[i]=xb2[i];
	}
	fclose(stream3);
}