#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <assert.h>
#include <cublas_v2.h>

void args_parser(int argc, char* argv[], double& acc, size_t& netSize, size_t& itCount) {
	if (argc < 4) {
		std::cout << "Options:\n\t-accuracy\n\t-netSize\n\t-itCount\n";
		std::cout << "Usage: transcalency [option]=[value]" << std::endl;
		exit(0);
	}
	bool specified[] = { false, false, false };
	std::string args[] = { "-accuracy", "-netSize", "-itCount" };
	
	for (int i = 1; i < argc; i++) {
		for (int j = 0; j < 3; j++) {
			std::string cmpstr(argv[i]);
			if (!specified[j] && cmpstr.rfind(args[j]) == 0) {
				specified[j] = true;
				double val;
				std::stringstream ss(cmpstr.substr(args[j].length() + 1));
				if (!(ss >> val)) {
					std::cerr << "Can't parse " << args[j] << std::endl;
					exit(1);
				}
				ss.flush();
				switch (j)
				{
				case 0:
					acc = val;
					break;
				case 1:
					netSize = val;
					if (val < 0) {
						std::cerr << "netSize can't be < 0" << std::endl;
						exit(1);
					}
					break;
				case 2:
					itCount = val;
					if (val < 0) {
						std::cerr << "itCount can't be < 0" << std::endl;
						exit(1);
					}
					break;
				default:
					std::cout << "unexpected option " << args[i] << "\n"; 
					break;
				}
				continue;
			}
		}
	}

	for (int i = 0; i < 3; i++) {
		if (!specified[i]) {
			std::cerr << "Option " << args[i] << " not specified" << std::endl;
			exit(1);
		}
	}
}

int main(int argc, char* argv[]) {
	double accuracy;
	size_t netSize=0, itCountMax;
	args_parser(argc, argv, accuracy, netSize, itCountMax);

	double loss = 0;
	int itCount = 0;

	size_t size = netSize*netSize;
    
    double* A = new double[size];
	double* Anew = new double[size];
	double* Delta = new double[size];

	memset(A, 0, sizeof(double)*size); //most values at init step should be zero

	//set values to corners
	A[0] = 10;
	A[netSize - 1] = 20;
	A[netSize*(netSize - 1)] = 30;
	A[netSize*netSize-1] = 20;

	//linear interpolation steps
	double hor_top_step = (A[netSize - 1] - A[0]) / (netSize - 1); 
	double hor_down_step = (A[netSize*netSize - 1] - A[netSize*(netSize-1)]) / (netSize - 1);
	double ver_left_step = (A[netSize * (netSize - 1)] - A[0]) / (netSize - 1);
	double ver_right_step = (A[netSize*netSize-1] - A[netSize-1]) / (netSize - 1);

	// set values to sides
	for(int i = 1; i < netSize - 1; i++) {
		A[i] = 10 + hor_top_step*i;
		A[netSize*i] = 10 + ver_left_step*i;
		A[netSize*(i+1) -1] = 20 + ver_right_step*i;
		A[netSize*(netSize-1) + i] = 30 + hor_down_step*i;
	}

	memcpy(Anew, A, sizeof(double)*size); // copy corners and sides to Anew
	cublasHandle_t handle;
	cublasCreate(&handle);
	#pragma acc data copyin(A[:size], Anew[:size], Delta[:size])
	{
		for(itCount = 0; itCount < itCountMax; itCount++)
		{
			#pragma acc data present(A[:size], Anew[:size]) // update pointers on gpu
			#pragma acc parallel loop 
			for(int y = 1; y < netSize - 1; y++) {
				#pragma acc loop
				for(int x = 1; x < netSize - 1; x++) {
					Anew[y*netSize + x] = 0.25 * (A[(y+1)*netSize + x] + A[(y-1)*netSize + x] + A[y*netSize + x + 1] + A[y*netSize + x - 1]);
				}
			}

			if(itCount%100 == 0 || itCount + 1 == itCountMax) { // calc loss every 100 iterations or last
				loss = 0;
				//#pragma acc data copy(loss)
				// #pragma acc parallel loop reduction(max:loss)
				// for(int y = 1; y < netSize - 1; y++) {
				// 	#pragma acc loop reduction(max:loss)
				// 	for(int x = 1; x < netSize - 1; x++) {
				// 		loss = std::fmax(loss, std::fabs(Anew[y*netSize+x] - A[y*netSize + x]));
				// 	}
				// }

				
				double a = -1;
				int id = 0;
				
				cublasStatus_t res1, res2, res3;
				#pragma acc host_data use_device(A, Anew, Delta)
				{
					res1 = cublasDcopy(handle, size, Anew, 1, Delta, 1);
					res2 = cublasDaxpy(handle, size, &a, A, 1, Delta, 1);
					res3 = cublasIdamax(handle, size, Delta, 1, &id);
				}

				if(res1 != CUBLAS_STATUS_SUCCESS)
					exit(EXIT_FAILURE);
				if(res2 != CUBLAS_STATUS_SUCCESS)
					exit(EXIT_FAILURE);
				if(res3 != CUBLAS_STATUS_SUCCESS)
					exit(EXIT_FAILURE);
				#pragma acc update host(Delta[id])	
				loss = std::abs(Delta[id]);
				
				if(loss <= accuracy) // finish calc if needed accuracy reached
					break;
			}
			std::swap(A, Anew); // swap pointers on cpu
		}
	}
	cublasDestroy(handle);
	std::cout << loss << '\n';
	std::cout << itCount << '\n';
	#pragma acc exit data delete(A[:size], Anew[:size], Delta[:size])
	delete[] A;
	delete[] Anew;
	delete[] Delta;
	return 0;
}
