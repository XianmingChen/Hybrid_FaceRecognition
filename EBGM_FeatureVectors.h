#include <string.h>
#include "GaborFilterResponse.h"

void copy_to_Point3D(double ***feature_vectors,double Each_feature_vector[][41][2],int n1,int n2,int n3);
void Point3D_copy(double EachImage_Filter_Respone[][Width][2],double ***Each_Gabor_Respone,int n1,int n2,int n3);
void EBGM_FeatureVectors(double Gabor_Respone_Mean_Value[][2],int *feature_count,double Each_Feature_Vectors[][41][2]);
