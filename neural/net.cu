#include <iostream>
#include <fstream>

float *w_fc1, *w_fc2, *w_fc3;

void allocate(int fc1, int fc2, int fc3, int fc4)
{
    cudaMalloc((void**)&w_fc1, sizeof(float)*fc1*fc2);
    cudaMalloc((void**)&w_fc2, sizeof(float)*fc2*fc3);
    cudaMalloc((void**)&w_fc3, sizeof(float)*fc3*fc4);
}

void loadWeights(const char* path)
{
    std::ifstream in(path);

}

void forward(float* input, float* output)
{
    
}

int main(int argc, char** argv)
{
    allocate(1024, 256, 16, 1);
    loadWeights(argv[2]);

    return 0;
}