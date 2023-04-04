#include <iostream>
#include <sstream>
#include <cmath>
#include <cub/cub.cuh>

using namespace cub;
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

void catchFailure(const std::string& operation_name) {
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << operation_name << " failed " << std::endl;
        exit(EXIT_FAILURE);
    }
}

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

__global__ void fillEdges(double* A, size_t netSize, double hor_top_step, double hor_down_step, double ver_left_step, double ver_right_step)
{
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    A[i] = 10 + hor_top_step*i;
    A[netSize*i] = 10 + ver_left_step*i;
    A[netSize*(i+1) -1] = 20 + ver_right_step*i;
    A[netSize*(netSize-1) + i] = 30 + hor_down_step*i;
}

__global__ void solve(const double *A, double *Anew, unsigned long long netSize)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x == 0 || y == 0 || x >= netSize -1 || y >= netSize -1)
        return;
    Anew[y*netSize + x] = 0.25 * (A[(y+1)*netSize + x] + A[(y-1)*netSize + x] + A[y*netSize + x + 1] + A[y*netSize + x - 1]);
}

__global__ void getDelta(double* Anew, double* A, double* Delta, int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size)
        return;
    Delta[x] = Anew[x] - A[x];
}

int main(int argc, char* argv[]) {
    double accuracy;
    size_t netSize=0, itCountMax;
    args_parser(argc, argv, accuracy, netSize, itCountMax);
    
    double loss = 0;
    int itCount;

    size_t size = netSize*netSize;

    double *A, *Anew, *Delta;
    cudaMalloc((void**)&A, sizeof(double)*size);
    cudaMalloc((void**)&Anew, sizeof(double)*size);
    cudaMalloc((void**)&Delta, sizeof(double)*size);
    catchFailure("alloc");
    cudaMemset(A, 0, sizeof(double)*size);

    //linear interpolation steps
    double hor_top_step = (double)(20 - 10) / (double)(netSize - 1);
    double hor_down_step = (double)(20 - 30) / (double)(netSize - 1);
    double ver_left_step = (double)(30 - 10) / (double)(netSize - 1);
    double ver_right_step = (double)(20 - 20) / (double)(netSize - 1);

    // set values to sides
    unsigned int block = netSize/32;
    if(netSize % 32 != 0)
        block+=1;
    fillEdges<<<block, 32>>>(A, netSize, hor_top_step, hor_down_step, ver_left_step, ver_right_step);
    catchFailure("fill edges");
    cudaMemcpy(Anew, A, sizeof(double)*size, cudaMemcpyDeviceToDevice); // copy corners and sides to Anew
    catchFailure("copy A to Anew");
    dim3 threads(32, 32, 1);
    dim3 blocks(netSize/32, netSize/32, 1);
    if(netSize % 32 != 0) {
        blocks.x += 1;
        blocks.y += 1;
    }
    double* val;
    cudaMalloc((void**)&val, sizeof(double));

    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Delta, val, size));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    for(itCount = 0; itCount < itCountMax; itCount++)
    {
        solve<<<blocks, threads>>>(A, Anew, netSize);
        if(itCount%100 == 0 || itCount + 1 == itCountMax) { // calc loss every 100 iterations or last
            int delta_blocks = size/1024;
            if(size%1024!=0)
                delta_blocks+=1;
            getDelta<<<delta_blocks, 1024>>>(Anew, A, Delta, size);
            // Run
            CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Delta, val, size));
            
            double h_val;
            cudaMemcpy(&h_val, val, sizeof(double), cudaMemcpyDeviceToHost);

            loss = std::abs(h_val);

            if(loss <= accuracy) // finish calc if needed accuracy reached
                break;
        }
        std::swap(A, Anew); // swap pointers on cpu
    }
    catchFailure("calc");
    if(netSize <= 32){
        double* results = new double[size];
        cudaMemcpy(results, A, sizeof(double)*size, cudaMemcpyDeviceToHost);
        for(int y = 0; y < netSize; y++){
            for(int x = 0; x < netSize; x++) {
                std::cout << results[y*netSize + x] << " ";
            }
            std::cout << std::endl;
        }
    }
    catchFailure("calc fail");
    std::cout << loss << '\n';
    std::cout << itCount << '\n';
    cudaFree(A);
    cudaFree(Anew);
    cudaFree(Delta);
    return 0;
}
