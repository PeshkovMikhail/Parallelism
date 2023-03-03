#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>

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
					if (webSize < 0) {
						std::cerr << "webSize can't be < 0" << std::endl;
						exit(1);
					}
					break;
				case 2:
					itCount = val;
					if (itCount < 0) {
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
	if (i + webSize > webSize*webSize) {
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
	size_t webSize, itCountMax;
	args_parser(argc, argv, accuracy, webSize, itCountMax);
	double loss = 0;
	size_t itCount;

	double* web = new double[webSize * webSize];
	memset(web, 0, sizeof(double) * webSize * webSize);

	web[0] = 10; // Top left
	web[webSize - 1] = 20; // Top right
	web[webSize * (webSize - 1)] = 30; // Down left
	web[webSize * webSize - 1] = 20; // Down right

	double hor_top_step = (web[webSize - 1] - web[0]) / (webSize - 1);
	double hor_down_step = (web[webSize*webSize - 1] - web[webSize*(webSize-1)]) / (webSize - 1);
	double ver_left_step = (web[webSize * (webSize - 1)] - web[0]) / (webSize - 1);
	double ver_right_step = (web[webSize*webSize-1] - web[webSize-1]) / (webSize - 1);

	for (int i = 1; i < webSize - 1; i++) {
		web[i] = web[0] + hor_top_step * i;
		web[webSize * (webSize - 1) + i] = web[webSize * (webSize - 1)] + hor_down_step * i;
		web[webSize*i] = web[0] + ver_left_step * i;
		web[webSize * (i+1) - 1] = web[webSize - 1] + ver_right_step * i;
	}
	double tempLoss = 0;
	for (itCount = 1; itCount <= itCountMax; itCount++) {
		for (int y = 0; y < webSize; y++) {
			for (int x = 0; x < webSize; x++) {
				double c = web[y * webSize + x];
				double avg = get_avg(web, y*webSize+x, webSize);
				if (std::abs(avg - c) > tempLoss)
					tempLoss = std::abs(avg - c);
				web[y * webSize + x] = avg;
			}
		}
		loss = tempLoss;
		if (accuracy >= loss)
			break;
		tempLoss = 0;
	}
	std::cout.precision(17);
	std::cout << "Loss: " << loss << "\n";
	std::cout << "Iterations: " << itCount << "\n";
	return 0;
}