#include "EBGM_FaceComparison.h"
#include "EBGM_FeatureVectors.h"

int EBGM_FaceComparison(int total_trainface,int train_feature_count[], double train_feature[][500][41][2],
						int probe_feature_count,double Probe_feature_Vectors[][41][2])
{
	int i,j,k,m;
	int MaxSimilarity_count[Total_train_face]={0};
	double OveralSimilarity[Total_train_face]={0.0};
	double temp_max_similarity=0.0;
	double point_peak_similarity=0.0;
	int max_similarity_index=0;
	int point_peak_index;
	int point_peak_existed=0;
	double similarity=0.0;
	double similarity_matrix[500][Total_train_face]={0.0};
	double temp_MaxSimilarity_count=0.0,temp_feature_count=0.0;

	double thresh_distance=8.0;
	double probe_xcoordiante;
	double probe_ycoordiante;
	double train_xcoordiante;
	double train_ycoordiante;
	double distance=0.0;
	double last_similarity=0.0;
	double dotproduct=0.0;
	double train_feature_sqr=0.0;
	double probe_feature_sqr=0.0;


	for (i=0;i<probe_feature_count;i++)
	{
		probe_xcoordiante=Probe_feature_Vectors[i][0][0];
		probe_ycoordiante=Probe_feature_Vectors[i][0][1];
		for (j=0;j<total_trainface;j++)
		{
			last_similarity=0.0;
			for (k=0;k<train_feature_count[j];k++)
			{
				similarity=0.0;
				train_xcoordiante=train_feature[j][k][0][0];
				train_ycoordiante=train_feature[j][k][0][1];
				distance=sqrt((train_xcoordiante-probe_xcoordiante)*(train_xcoordiante-probe_xcoordiante)+(train_ycoordiante-probe_ycoordiante)*(train_ycoordiante-probe_ycoordiante));
				if (distance<=thresh_distance)
				{
					dotproduct=0.0;
					train_feature_sqr=0.0;
					probe_feature_sqr=0.0;
					for (m=1;m<41;m++)  //the first one is position information
					{
						dotproduct=dotproduct+sqrt(complex_modulus(train_feature[j][k][m]))*sqrt(complex_modulus(Probe_feature_Vectors[i][m]));
						train_feature_sqr = train_feature_sqr + complex_modulus(train_feature[j][k][m]);
						probe_feature_sqr = probe_feature_sqr + complex_modulus(Probe_feature_Vectors[i][m]);
					}
					similarity=dotproduct/sqrt(train_feature_sqr*probe_feature_sqr);
					if (similarity>last_similarity)
					{
						similarity_matrix[i][j]=similarity;
						last_similarity=similarity;
					}
				}
			}
		}
	}

	for (i=0;i<probe_feature_count;i++)
	{
		point_peak_similarity=0.0;
		point_peak_existed=0;
		for (j=0;j<total_trainface;j++)
		{
			if (similarity_matrix[i][j]>point_peak_similarity)
			{
				point_peak_similarity=similarity_matrix[i][j];
				point_peak_index=j;
				point_peak_existed++;
			}
		}
		if (point_peak_existed>0)
		{
			MaxSimilarity_count[point_peak_index]++;
		}
	}

	for (i=0;i<total_trainface;i++)
	{
		temp_MaxSimilarity_count=MaxSimilarity_count[i];
		temp_feature_count=train_feature_count[i];
		OveralSimilarity[i]= temp_MaxSimilarity_count/temp_feature_count;
		if (temp_max_similarity<=OveralSimilarity[i])
		{
			temp_max_similarity=OveralSimilarity[i];
			max_similarity_index=i;
		}
	}
	return max_similarity_index;
}