#include "GaborFilterResponse.h"


double complex_modulus(double *t)
{
	return (t[0]*t[0]+t[1]*t[1]);
}

void convolv2_same(double x[][Width],double (*y)[Width][2],double (*z)[Width][2])
{
	int i,j,n,m;
	int r1,r2,c1,c2;
	int Start_Row,Start_Column;
	int Window_R, Window_C;
	double conv_full[Height*2-1][Width*2-1][2]={0.0,0.0};
	
	Window_R=(Height/2)-10; //20*20 window to convolution
	Window_C=(Width/2)-10;
	Start_Row=(Height/2);
	Start_Column=(Width/2);

	for (i=0;i<Height;i++)
	{
		for (j=0;j<Width;j++)
		{
			r1=i;
			r2=r1+Height;
			c1=j;
			c2=c1+Width;
			for (m=r1+Window_R;m<r2-(Window_R+1);m++)
			{
				for (n=c1+Window_C;n<c2-Window_C;n++)
				{
					conv_full[m][n][0]=conv_full[m][n][0]+x[i][j]*y[m-r1][n-c1][0];
					conv_full[m][n][1]=conv_full[m][n][1]+x[i][j]*y[m-r1][n-c1][1];
				}
			}
		}
	}

	for (i=0;i<Height;i++)
	{
		for (j=0;j<Width;j++)
		{
			z[i][j][0]=conv_full[i+Start_Row][j+Start_Column][0];
			z[i][j][1]=conv_full[i+Start_Row][j+Start_Column][1];
		}
	}
}

void GaborWavelet(int Row,int Column,double Kmax,double f,int u,int v,double Delt_sqr,double GW[][Width][2])
{
	int m,n;
	double k[2];
	double abs_k,abs_k_sqr;
	abs_k=Kmax/pow(f,v);
	abs_k_sqr=pow(abs_k,2);
	k[0]=abs_k*cos(u*PI/8);
	k[1]=abs_k*sin(u*PI/8);

	for (m=0;m<Row;m++)
	{
		for (n=0;n<Column;n++)
		{
			GW[m][n][0]=(abs_k_sqr/Delt_sqr)*exp(-0.5*abs_k_sqr*(pow((m-Row/2+0.5),2)+pow((n-Column/2+1),2))/Delt_sqr)*(cos(k[0]*(m-Row/2+0.5)+k[1]*(n-Column/2+1))-exp(-0.5*Delt_sqr));
			GW[m][n][1]=(abs_k_sqr/Delt_sqr)*exp(-0.5*abs_k_sqr*(pow((m-Row/2+0.5),2)+pow((n-Column/2+1),2))/Delt_sqr)*sin(k[0]*(m-Row/2+0.5)+k[1]*(n-Column/2+1));
		} 
	}
}

void Gabor_Respone_Mean(double *temp_mean,double Convolv[Height][Width][2])
{
	int i,j;
	double temp_real=0.0;
	double temp_image=0.0;
	for (i=0;i<Height;i++)
	{
		for (j=0;j<Width;j++)
		{
			temp_real=temp_real+Convolv[i][j][0];
			temp_image=temp_image+Convolv[i][j][1];
		}
	}
	temp_mean[0]=temp_real/(Height*Width);
	temp_mean[1]=temp_image/(Height*Width);
}

void GaborFilterResponse(double trainface[][Width],double Mean_Value[][2])
{
	int v,u;
	int j,k;
	double Kmax=PI/2;
	double f=sqrt(0.7);
	double Delt = 1.5 * PI;
	double Delt_sqr = Delt * Delt;
	int gf_filter_count = 5*8;
	double Gabor_Wavelet[Height][Width][2]={0.0};
	double Convolv[Height][Width][2]={0.0};
	double temp_mean[2]={0.0};

	for (v=0;v<5;v++)
	{
		for (u=1;u<9;u++)
		{
			GaborWavelet(Height,Width,Kmax,f,u,v,Delt_sqr,Gabor_Wavelet);
			convolv2_same(trainface,Gabor_Wavelet,Convolv);
			for (j=0;j<Height;j++)
			{
				for (k=0;k<Width;k++)
				{
					Gabor_Respone[v*8+u-1][j][k][0]=Convolv[j][k][0];
					Gabor_Respone[v*8+u-1][j][k][1]=Convolv[j][k][1];
				}
			}
			Gabor_Respone_Mean(temp_mean,Convolv);
			Mean_Value[v*8+u-1][0]=temp_mean[0];
			Mean_Value[v*8+u-1][1]=temp_mean[1];
		} 
	}
}