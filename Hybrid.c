#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include "EBGM_FeatureVectors.h"
#include "EBGM_FaceComparison.h"
double Gabor_Respone[Filter_Num][Height][Width][2];
double Feature_Vectors[Total_train_face][500][41][2];
#define candidate_num 15


void Exchange(float *A, float *B)
{
	float temp;
	temp=*A;
	*A=*B;
	*B=temp;
}

int random_num(int start, int end)
{
	int num;

	srand((unsigned)time(NULL)); 
	num=(rand()%(end-start))+start;
	return num;
}

int partition(float *Address, int start, int end)
{
	float pivot;
	int i,j;

	i=start-1;
	pivot=Address[end];

	for (j=start;j<end;j++)
	{
		if (Address[j]<pivot)
		{
			i++;
			Exchange(&Address[i],&Address[j]);
		}
	}
	Exchange(&Address[end],&Address[i+1]);
	return i+1;
}

int randomized_partition(float *Address, int start, int end)
{
	int random_position;

	random_position=random_num(start,end);
	Exchange(&Address[random_position],&Address[end]);
	return partition(Address,start,end);
}

float randomized_selection(float *Address, int start, int end, int position)
{
	int random_position;
	int temp_rank;

	if (start==end)
		return Address[start];
	random_position=randomized_partition(Address,start,end);
	temp_rank=random_position-start+1;

	if (temp_rank>position+1)
		return randomized_selection(Address,start,random_position-1,position);
	else if (temp_rank<position+1)
		return randomized_selection(Address,random_position+1,end,position-temp_rank);
	else //if (pivot==position)
		return Address[random_position];
}

int search_index(float *Address, int length, float target_value)
{
	int i;
	for (i=0;i<length;i++)
	{
		if (target_value==Address[i])
		{
			return i;
		}
	}
	return -1;
}

void read_image(char *filepath,double image[][Width])
{
	int i,j;
	CvScalar s;
	IplImage *img=cvLoadImage(filepath,0);
    for(i=0;i<img->height;i++)
	{
        for(j=0;j<img->width;j++)
		{
			s=cvGet2D(img,i,j); 
			image[i][j]=s.val[0]/255;
        }
    }
    cvReleaseImage(&img);  
}

void PCA_load_image(char *filepath,CvMat *input,int rank)
{
	int i;
	float *ptr,*ptr2;
	IplImage *img;
	CvMat row_header,*row,*mat;

	img=cvLoadImage(filepath,0);
	mat=cvCreateMat(Height,Width,CV_32FC1);
	cvConvert(img,mat);
	row=cvReshape(mat,&row_header,0,1);

	ptr=(float*)(input->data.fl+rank*input->cols); //input->cols=Height*Width
	ptr2=(float*)row->data.fl;

	for (i=0;i<input->cols;i++)
	{
		*ptr=*ptr2;
		ptr++;
		ptr2++;
	}
	cvReleaseImage(&img);  
}


void PCA_Comparison(CvMat* training,CvMat* probing,int candidate_index[][candidate_num],int rank)
{
	int i,j;
	int temp_index;
	float difference[Total_train_face]={0.0};
	float copy_difference[Total_train_face]={0.0};
	float temp_difference;
	float* temp_train;
	float* temp_probe;

	for (i=0;i<training->rows;i++)
	{
		difference[i]=0.0;
		temp_train=(float*)training->data.fl+i*training->cols;
		temp_probe=(float*)probing->data.fl;
		for (j=0;j<probing->cols;j++)
		{
			difference[i]=(float)(*temp_train-*temp_probe)*(*temp_train-*temp_probe)+difference[i];
			copy_difference[i]=difference[i]; 
			temp_train++;
			temp_probe++;
		}
	}

	for (i=0;i<candidate_num;i++)
	{
		temp_difference=randomized_selection(copy_difference,0,training->rows-1,i);

		temp_index=search_index(difference,training->rows,temp_difference);
		candidate_index[rank][i]=temp_index;
	}
}

void PCA_process(int candidate_index[][candidate_num])
{
	char image_path[255];
	int i;
	CvMat *training_space,*training_result,*probing_space,*probing_result;
	CvMat *average,*eigen_values,*eigen_vectors;

	printf("PCA preprocessing...\n");
	training_space=cvCreateMat(Total_train_face,Height*Width,CV_32FC1);
	for (i=0;i<Total_train_face;i++)
	{
		printf("Reading image %d...\n",i+1);
		sprintf(image_path,"Aligned_FERET/input/trainfaces/%d.jpg",i+1);
		PCA_load_image(image_path,training_space,i);
	}

	printf("Begin to PCA train images...\n");
	average=cvCreateMat(1,training_space->cols,CV_32FC1);
	eigen_values=cvCreateMat(1,min(training_space->rows,training_space->cols),CV_32FC1);
	eigen_vectors=cvCreateMat(min(training_space->rows,training_space->cols),training_space->cols,CV_32FC1);
	cvCalcPCA(training_space,average,eigen_values,eigen_vectors,CV_PCA_DATA_AS_ROW);

	training_result=cvCreateMat(training_space->rows,min(training_space->rows,training_space->cols),CV_32FC1);
	cvProjectPCA(training_space,average,eigen_vectors,training_result);

	for (i=0;i<Total_probe_face;i++)
	{
		probing_space=cvCreateMat(1,Height*Width,CV_32FC1);
		probing_result=cvCreateMat(probing_space->rows,min(training_space->rows,training_space->cols),CV_32FC1);

		sprintf(image_path,"Aligned_FERET/input/probefaces/%d.jpg",i+1);
		PCA_load_image(image_path,probing_space,0);
		cvProjectPCA(probing_space,average,eigen_vectors,probing_result);
		
		PCA_Comparison(training_result,probing_result,candidate_index,i);

		cvReleaseMat(&probing_space);
		cvReleaseMat(&probing_result);
	}
	cvReleaseMat(&average);
	cvReleaseMat(&eigen_values);
	cvReleaseMat(&eigen_vectors);

	cvReleaseMat(&training_space);
	cvReleaseMat(&training_result);
}

double EBGM(int candidate_index[][candidate_num])
{
	char image_path[255];
	int i,j;
	double trainface[Height][Width]={0.0};

	double Mean_Value[Filter_Num][2]={0.0};
	double Each_Feature_Vectors[500][41][2]={0.0};

	int train_feature_count[Total_train_face]={0};
	int each_feature_count=0;
	int probe_feature_count=0;

	int return_index;
	int real_index;
	int active_train_feature_count[candidate_num]={0};
	double active_train_Feature[candidate_num][500][41][2];

	int Probe_count=0;
	double accuracy;

	for (i=0;i<Total_train_face;i++)
	{
		printf("EBGM Training image %d...\n",i+1);
		sprintf(image_path,"Aligned_FERET/input/trainfaces/%d.jpg",i+1);
		read_image(image_path,trainface);
		GaborFilterResponse(trainface,Mean_Value);
		EBGM_FeatureVectors(Mean_Value,&each_feature_count,Each_Feature_Vectors);
		train_feature_count[i]=each_feature_count;
		each_feature_count=0;
		memcpy(Feature_Vectors[i],Each_Feature_Vectors,500*41*2*8);
	}

	for (i=0;i<Total_probe_face;i++)
	{
		printf("Image %d Probing...\n",i+1);
		sprintf(image_path,"Aligned_FERET/input/probefaces/%d.jpg",i+1);
		read_image(image_path,trainface);
		GaborFilterResponse(trainface,Mean_Value);
		EBGM_FeatureVectors(Mean_Value,&each_feature_count,Each_Feature_Vectors);
		probe_feature_count=each_feature_count;
		each_feature_count=0;
		
		for (j=0;j<candidate_num;j++)
		{
			real_index=candidate_index[i][j];
			active_train_feature_count[j]=train_feature_count[real_index];
			memcpy(active_train_Feature[j],Feature_Vectors[real_index],500*41*2*8);
		}

		return_index=EBGM_FaceComparison(candidate_num,train_feature_count,active_train_Feature,
									probe_feature_count,Each_Feature_Vectors);

		if (candidate_index[i][return_index]==i)
		{
			Probe_count++;
			printf("Image %d Probe successfully!\n",i+1);
		}
		else
		{
			 printf("Image %d Probe failed!\n",i+1);
		}
	}
	accuracy=(double)Probe_count/Total_probe_face;
	printf("Probe accuary of %d images in hybrid solution is:= %f\n",Total_probe_face,accuracy);
	return accuracy;
}


int main()
{
	int candidate_index[Total_probe_face][candidate_num];
	double accuracy;
	FILE *record;
	double start,end;

	start=clock();
	record=fopen("record_Hybrid.txt","a+");

	PCA_process(candidate_index);
	accuracy=EBGM(candidate_index);

	end=clock();
	start=end-start;
	fprintf(record,"The total running time is %f ms.\n",start);
	fprintf(record,"Probe accuary of %d images is:= %f\n",Total_probe_face,accuracy);
	fprintf(record,"\n");
	fclose(record);

	system("pause");
	return 0;
}
