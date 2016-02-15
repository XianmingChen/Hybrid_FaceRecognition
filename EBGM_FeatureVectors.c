#include "EBGM_FeatureVectors.h"

void find_peak(double EachImage_Filter_Respone[][Width][2],int row_start,int row_end,int column_start,int column_end,int *row_position, int *column_position)
{
	double temp=0.0;
	double local_max=0.0;
	int i,j;
	for (i=row_start;i<row_end;i++)
	{
		for (j=column_start;j<column_end;j++)
		{
			temp=complex_modulus(EachImage_Filter_Respone[i][j]);
			if (local_max<=temp)
			{
				local_max=temp;
				*row_position=i;
				*column_position=j;
			}
		}
	}
}

void copy_from_Point3D(double EachImage_Filter_Respone[][Width][2],double ***Each_Gabor_Respone,int n1,int n2,int n3)
{
	int i,j,k;
	for (i=0;i<n1;i++)
	{
		for (j=0;j<n2;j++)
		{
			for (k=0;k<n3;k++)
			{
				EachImage_Filter_Respone[i][j][k]=Each_Gabor_Respone[i][j][k];
			}
		}
	}
}

void copy_to_Point3D(double ***feature_vectors,double Each_feature_vector[][41][2],int n1,int n2,int n3)
{
	int i,j,k;
	for (i=0;i<n1;i++)
	{
		for (j=0;j<n2;j++)
		{
			for (k=0;k<n3;k++)
			{
				feature_vectors[i][j][k]=Each_feature_vector[i][j][k];
			}
		}
	}
}


void EBGM_FeatureVectors(double Gabor_Respone_Mean_Value[][2],int *feature_count,double Each_Feature_Vectors[][41][2])
{
	int j,k,m;
	int row_index,column_index;
	int Row_Window=20;
	int Column_Window=20;
	int Threshold=2;
	double EachImage_Filter_Respone[Height][Width][2]={0.0};
	double local_peak=0.0;
	char Gabor_peak_count[Height][Width]={0};
	int row_position; 
	int column_position;
	int count=0;

	for (j=0;j<Filter_Num;j++)
	{
		memcpy(EachImage_Filter_Respone,Gabor_Respone[j],Height*Width*2*8);
		for (row_index=0;row_index<(int)Height/Row_Window+1;row_index++)
		{
			for (column_index=0;column_index<(int)Width/Column_Window+1;column_index++)
			{
				if (((row_index+1)*Row_Window<Height)&&((column_index+1)*Column_Window<Width))
				{
					find_peak(EachImage_Filter_Respone,row_index*Row_Window,(row_index+1)*Row_Window,column_index*Column_Window,(column_index+1)*Column_Window,&row_position,&column_position);
				}
				else if(((row_index+1)*Row_Window>Height)&&((column_index+1)*Column_Window<Width))
				{
					find_peak(EachImage_Filter_Respone,row_index*Row_Window,Height,column_index*Column_Window,(column_index+1)*Column_Window,&row_position,&column_position);
				}
				else if(((row_index+1)*Row_Window<Height)&&((column_index+1)*Column_Window>Width))
				{
					find_peak(EachImage_Filter_Respone,row_index*Row_Window,(row_index+1)*Row_Window,column_index*Column_Window,Width,&row_position,&column_position);
				}
				else if(((row_index+1)*Row_Window>Height)&&((column_index+1)*Column_Window>Width))
				{
					find_peak(EachImage_Filter_Respone,row_index*Row_Window,Height,column_index*Column_Window,Width,&row_position,&column_position);
				}
				
				if (EachImage_Filter_Respone[row_position][column_position][0]>Gabor_Respone_Mean_Value[j][0])
				{
					Gabor_peak_count[row_position][column_position]++;
				}
			}
		}

	}
	
	count=(*feature_count);
	for (j=0;j<Height;j++)
	{
		for (k=0;k<Width;k++)
		{
			if (Gabor_peak_count[j][k]>Threshold)
			{
				for (m=0;m<41;m++)
				{
					if (m==0)
					{
						Each_Feature_Vectors[count][0][0]=j;
						Each_Feature_Vectors[count][0][1]=k;
					}
					else
					{
						Each_Feature_Vectors[count][m][0]=Gabor_Respone[m-1][j][k][0];
						Each_Feature_Vectors[count][m][1]=Gabor_Respone[m-1][j][k][1];
					}
				}
				count++;  
			}
			else
			{
				Gabor_peak_count[j][k]=0;
			}
		}
	}
	*feature_count=count;
}