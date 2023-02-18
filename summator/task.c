#define LENGTH 10000000
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main() {
	double* arr = (double*)malloc(sizeof(double)*LENGTH);
	if(arr == NULL)
		return -1;
	clock_t start = clock();
	double step = 2*M_PI/LENGTH;
	#pragma acc enter data create(arr[:LENGTH]/*, arr_d[:LENGTH]*/)
	#pragma acc parallel loop present(arr[:LENGTH]) 
	for(size_t i = 0; i < LENGTH; i++)
		arr[i] = sin(i*step);
	printf("%f\n", (float)(clock() - start)/CLOCKS_PER_SEC);
	double sum = 0;
	#pragma acc parallel loop present(arr)
	for(size_t i = 0; i < LENGTH; i++)
		sum += arr[i];
	printf("%f\n", (float)(clock() - start)/CLOCKS_PER_SEC);

	printf("%.32lf\n", sum);
	free(arr);
	//free(arr_d);
	return 0;
}
		
