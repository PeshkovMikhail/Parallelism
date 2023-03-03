#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <assert.h>

void args_parser(int argc, char* argv[], double& acc, size_t& webSize, size_t& itCount) {
	if (argc < 4) {
		std::cout << "Options:\n\t-accuracy\n\t-webSize\n\t-itCount\n";
		std::cout << "Usage: transcalency [option]=[value]" << std::endl;
		exit(0);
	}
	bool specified[] = { false, false, false };
	std::string args[] = { "-accuracy", "-webSize", "-itCount" };
	
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
					acc = std::max(val, std::pow(10, -6));
					break;
				case 1:
					webSize = val;
					if (val < 0) {
						std::cerr << "webSize can't be < 0" << std::endl;
						exit(1);
					}
					break;
				case 2:
					itCount = std::min(val, std::pow(10, 6));
					if (val < 0) {
						std::cerr << "itCount can't be < 0" << std::endl;
						exit(1);
					}
					break;
				default:
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

inline double get_avg(double* webPrev, int i, int webSize, int arr_num){
	int count = 1; // count of neighbours
    int pos = arr_num*webSize*webSize+i;

	double sum = webPrev[pos];
	if(i - (int)webSize >= 0){
		sum += webPrev[pos-webSize];
		count++;
	}
	if (i + webSize < webSize*webSize) {
		count++;
		sum += webPrev[pos+webSize];
	}
	if (i%webSize != 0) {
		count++;
		sum += webPrev[pos-1];
	}
	if (i%webSize != webSize-1) {
		count++;
		sum += webPrev[pos+1];
	}
	return sum/count;
}

int main(int argc, char* argv[]) {
	double accuracy;
	size_t webSize=0, itCountMax;
	args_parser(argc, argv, accuracy, webSize, itCountMax);

	double loss = 0;
	int itCount = 0;
    
    size_t size = webSize*webSize*2;
	double* web = new double[size];
	/*
	double sized array for storing previous and actual values. First webSize^2 previos values then webSize^2 actual values 
	*/
	memset(web, 0, sizeof(double) * size);

	web[0] = 10; // Top left
	web[webSize - 1] = 20; // Top right
	web[webSize * (webSize - 1)] = 30; // Down left
	web[webSize * webSize - 1] = 20; // Down right

	double hor_top_step = (web[webSize - 1] - web[0]) / (webSize - 1);
	double hor_down_step = (web[webSize*webSize - 1] - web[webSize*(webSize-1)]) / (webSize - 1);
	double ver_left_step = (web[webSize * (webSize - 1)] - web[0]) / (webSize - 1);
	double ver_right_step = (web[webSize*webSize-1] - web[webSize-1]) / (webSize - 1);
	
    int halfSize = size/2;
	#pragma acc data copy(web[0:size])
	{
		#pragma acc parallel loop
		for (int i = 1; i < webSize - 1; i++) {
			web[i] = web[0] + hor_top_step * i; // top left corner
			web[webSize * (webSize - 1) + i] = web[webSize * (webSize - 1)] + hor_down_step * i; // top right corner
			web[webSize*i] = web[0] + ver_left_step * i; // down left corner
			web[webSize * (i+1) - 1] = web[webSize - 1] + ver_right_step * i; // down right corner
		}

		for (itCount = 0; itCount < itCountMax; itCount++) {
			loss = 0;
            int offset = (itCount%2)*size/2;
            int next_offset = ((itCount+1)%2)*size/2;
			// swapping buffer pointers on gpu is not available. 

			#pragma acc parallel loop
			for( int i = 0; i < size/2; i++ ) {
				web[next_offset + i] = get_avg(web, i, webSize, itCount%2);
			}

			#pragma acc parallel loop reduction(max:loss)
			for( int i = 0; i < size/2; i++ ) {
				loss =  fmax(loss,fabs(web[next_offset+i] - web[offset+i]));
			}

			if (accuracy >= loss)
				break;
		}
	}

	std::cout << "Loss: " << loss << "\n";
	std::cout << "Iterations: " << itCount << "\n";
	#pragma acc exit data delete(web[:size])
	delete[] web;
	return 0;
}