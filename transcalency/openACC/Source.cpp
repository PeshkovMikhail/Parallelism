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
					acc = val;
					break;
				case 1:
					webSize = val;
					if (val < 0) {
						std::cerr << "webSize can't be < 0" << std::endl;
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

inline double get_avg(double* webPrev, int i, int webSize){
	int count = 1;
	double sum = webPrev[i];
	if(i - (int)webSize >= 0){
		sum += webPrev[i-webSize];
		count++;
	}
	if (i + webSize < webSize*webSize) {
		count++;
		sum += webPrev[i+webSize];
	}
	if (i%webSize != 0) {
		count++;
		sum += webPrev[i-1];
	}
	if (i%webSize != webSize-1) {
		count++;
		sum += webPrev[i+1];
	}
	return sum/count;
}

int main(int argc, char* argv[]) {
	double accuracy;
	size_t webSize=0, itCountMax;
	args_parser(argc, argv, accuracy, webSize, itCountMax);
	double loss = 0;
	size_t itCount;

	double* web = new double[webSize * webSize];
	double* webPrev = new double[webSize * webSize];
	memset(webPrev, 0, sizeof(double) * webSize * webSize);
	webPrev[0] = 10; // Top left
	webPrev[webSize - 1] = 20; // Top right
	webPrev[webSize * (webSize - 1)] = 30; // Down left
	webPrev[webSize * webSize - 1] = 20; // Down right

	double hor_top_step = (webPrev[webSize - 1] - webPrev[0]) / (webSize - 1);
	double hor_down_step = (webPrev[webSize*webSize - 1] - webPrev[webSize*(webSize-1)]) / (webSize - 1);
	double ver_left_step = (webPrev[webSize * (webSize - 1)] - webPrev[0]) / (webSize - 1);
	double ver_right_step = (webPrev[webSize*webSize-1] - webPrev[webSize-1]) / (webSize - 1);


	int size = webSize*webSize;
	#pragma acc data create(web[0:size]), copyin(webPrev[0:size])
	{
		#pragma acc parallel loop
		for (int i = 1; i < webSize - 1; i++) {
			webPrev[i] = webPrev[0] + hor_top_step * i;
			webPrev[webSize * (webSize - 1) + i] = webPrev[webSize * (webSize - 1)] + hor_down_step * i;
			webPrev[webSize*i] = webPrev[0] + ver_left_step * i;
			webPrev[webSize * (i+1) - 1] = webPrev[webSize - 1] + ver_right_step * i;
		}

		for (itCount = 0; itCount < itCountMax; itCount++) {
			loss = 0;
			#pragma acc parallel loop
			for( int i = 0; i < size; i++ )
			{
				web[i] = get_avg(webPrev, i, webSize);
			}

			#pragma acc parallel loop reduction(max:loss)
			for( int i = 0; i < size; i++ )
			{
				loss =  fmax(loss,fabs(web[i] - webPrev[i]));
			}

			if (accuracy >= loss)
				break;

			std::swap(webPrev,web);
		}
	}
	std::cout.precision(17);
	std::cout << "Loss: " << loss << "\n";
	std::cout << "Iterations: " << itCount << "\n";
	#pragma acc exit data delete(webPrev[:size], web[:size])
	delete[] web;
	delete[] webPrev;
	return 0;
}