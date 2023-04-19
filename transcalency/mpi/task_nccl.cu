#include <iostream>
#include <sstream>
#include <cmath>
#include <cub/cub.cuh>
#include <mpi.h>
#include <nccl.h>

#define NCCLCHECK(cmd) do {                               \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
      printf("Failed, NCCL error %s:%d '%s'\n",       \
          __FILE__,__LINE__,ncclGetErrorString(r));   \
      exit(EXIT_FAILURE);                             \
    }}                                                \
    while(0)

using namespace cub;
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

typedef struct {
    int basic;
    int first;
    int middle;
    int last;
    int self;
} Height;

typedef struct {
    int rank;
    int count;
    int netSize;
    int size;
    cudaStream_t stream;
    ncclComm_t comm;
    Height height;
} Context;

void catchFailure(const std::string& operation_name, cudaStream_t stream) {
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess || cudaStreamSynchronize(stream)) {
        std::cerr << operation_name << " failed " << std::endl;
        exit(EXIT_FAILURE);
    }
}

void args_parser(int argc, char* argv[], double& acc, int& netSize, int& itCount) {
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

__global__ void solve(const double *A, double *Anew, int netSize, int heightLimit)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x == 0 || y == 0 || x >= netSize -1 || y >=heightLimit -1)
        return;
    Anew[y*netSize + x] = 0.25 * (A[(y+1)*netSize + x] + A[(y-1)*netSize + x] + A[y*netSize + x + 1] + A[y*netSize + x - 1]);
}

__global__ void getDelta(double* Anew, double* A, double* Delta, int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size)
        return;
    Delta[x] = std::abs(Anew[x] - A[x]);
}


void shareLimits(double* A, Context ctx)
{
    if (ctx.rank!=ctx.count -1)
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(&A[(ctx.height.self-2)*ctx.netSize],ctx.netSize,ncclDouble,ctx.rank+1,ctx.comm, ctx.stream));
        NCCLCHECK(ncclRecv(&A[(ctx.height.self-1)*ctx.netSize],ctx.netSize,ncclDouble,ctx.rank+1,ctx.comm, ctx.stream));
        NCCLCHECK(ncclGroupEnd());
    }
    
    if (ctx.rank!=0)
    {
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(&A[ctx.netSize],ctx.netSize,ncclDouble,ctx.rank-1,ctx.comm, ctx.stream));
        NCCLCHECK(ncclRecv(A,ctx.netSize,ncclDouble,ctx.rank-1,ctx.comm, ctx.stream));
        NCCLCHECK(ncclGroupEnd());
    }    
}

void checkAccuracy(double* Anew, double* A, double* Delta, Context ctx, double accuracy, double* loss, bool* running, void *d_temp_storage, size_t temp_storage_bytes, double* val)
{
    int delta_blocks = ctx.size/1024;
    if(ctx.size%1024!=0)
        delta_blocks+=1;
    getDelta<<<delta_blocks, 1024, 0, ctx.stream>>>(Anew, A, Delta, ctx.size);
    // Run
    CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Delta, val, ctx.size, ctx.stream));
    
    cudaMemcpyAsync(loss, val, sizeof(double), cudaMemcpyDeviceToHost, ctx.stream);
    catchFailure("reduce", ctx.stream);

    
    if(ctx.rank != 0){
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclSend(val, 1, ncclDouble, 0, ctx.comm, ctx.stream));
        NCCLCHECK(ncclGroupEnd());
        catchFailure("send loss", ctx.stream);

        MPI_Recv(running, 1, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else{
        double temp_loss = 0;
        for(int i = 1; i < ctx.count; i++) {
            NCCLCHECK(ncclGroupStart());
            NCCLCHECK(ncclRecv(val, 1, ncclDouble, i, ctx.comm, ctx.stream));
            NCCLCHECK(ncclGroupEnd());
            cudaMemcpyAsync(&temp_loss, val, sizeof(double), cudaMemcpyDeviceToHost, ctx.stream);
            catchFailure("recv loss", ctx.stream);
            *loss = std::max(*loss, temp_loss);
        }

        if(*loss <= accuracy) // finish calc if needed accuracy reached
            *running = false;

        for(int i = 1; i < ctx.count; i++ )
        {
            MPI_Send(running, 1, MPI_C_BOOL, i, 0, MPI_COMM_WORLD);
        }
    } 
    
    catchFailure("check acc", ctx.stream);
}

void showResults(double* A_h, double* A, int size, int netSize)
{
    cudaMemcpy(A_h, A, sizeof(double)*size, cudaMemcpyDeviceToHost);
    for(int y = 0; y < netSize; y++){
        for(int x = 0; x < netSize; x++) {
            std::cout << A_h[y*netSize + x] << " ";
        }
        std::cout << std::endl;
    }
}

void selectDevice(int rank, int count)
{
    if(count > 4){
        if(rank == 0)
            std::cerr << "too much devices" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(cudaSetDevice(rank) != cudaSuccess){
        std::cerr << "device " << rank << "not available" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

Height getHeight(int rank, int count, int netSize)
{
    int basic = netSize/count;
    int middle = basic+2;
    int first = middle-1;
    int last = first + netSize%count;


    if(count == 1)
        first = netSize;
    int self = middle;
    if(rank == 0)
        self = first;
    else if(rank == count - 1)
        self = last;
    
    return {basic, first, middle, last, self};
}

Context getContext(int rank, int count, int netSize){
    Height height = getHeight(rank, count, netSize);
    int size = netSize*height.self;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, count, id, rank);

    return {rank, count, netSize, size, stream, comm, height};
}

void bufferInitialize(double*& A, double*& Anew, double*& Delta, double*& A_h, Context ctx)
{
    cudaMalloc((void**)&A, sizeof(double)*ctx.size);
    cudaMalloc((void**)&Anew, sizeof(double)*ctx.size);
    cudaMalloc((void**)&Delta, sizeof(double)*ctx.size);
    catchFailure("alloc", 0);
    cudaMemset(A, 0, sizeof(double)*ctx.size);

    //linear interpolation steps
    double hor_top_step = (double)(20 - 10) / (double)(ctx.netSize - 1);
    double hor_down_step = (double)(20 - 30) / (double)(ctx.netSize - 1);
    double ver_left_step = (double)(30 - 10) / (double)(ctx.netSize - 1);
    double ver_right_step = (double)(20 - 20) / (double)(ctx.netSize - 1);
    
    if(ctx.rank == 0)
    {
        size_t A_h_size = ctx.netSize * ctx.netSize;
        A_h = new double[A_h_size];
        std::memset(A_h, 0, sizeof(double)*A_h_size);
        for(int i = 0; i < ctx.netSize; i++)
        {
            A_h[i] = 10 + hor_top_step*i;
            A_h[ctx.netSize*i] = 10 + ver_left_step*i;
            A_h[ctx.netSize*(i+1) -1] = 20 + ver_right_step*i;
            A_h[ctx.netSize*(ctx.netSize-1) + i] = 30 + hor_down_step*i;
        }
        cudaMemcpy(A, A_h, sizeof(double)*ctx.size, cudaMemcpyHostToDevice);
        for(int i = 1 ; i < ctx.count; i++)
        {
            if(i == 0){
                MPI_Send(A_h, ctx.netSize*ctx.height.first, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            else if(i == ctx.count - 1) {
                MPI_Send(&A_h[ctx.netSize*ctx.height.basic*i - ctx.netSize], ctx.netSize*ctx.height.last, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            else
                MPI_Send(&A_h[ctx.netSize*ctx.height.basic*i - ctx.netSize], ctx.netSize*ctx.height.middle, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);    
        }        
    }
    else{
        A_h = new double[ctx.size];
        MPI_Recv(A_h, ctx.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaMemcpy(A, A_h, sizeof(double)*ctx.size, cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();
    // set values to sides
    cudaMemcpy(Anew, A, sizeof(double)*ctx.size, cudaMemcpyDeviceToDevice); // copy corners and sides to Anew
    catchFailure("copy A to Anew", 0);
    
}

void reductionInitialize(double*& val, double* Delta, void*& d_temp_storage, size_t& temp_storage_bytes, int size)
{
    cudaMalloc((void**)&val, sizeof(double));
    CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Delta, val, size));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
}


void getResultsAndShow(double* A_h, double* A, Context ctx)
{
    if(ctx.rank == 0){
        for(int i = 1; i < ctx.count; i++)
        {
            if(i == ctx.count - 1)
                MPI_Recv(&A_h[ctx.netSize*i*ctx.height.basic - ctx.netSize], ctx.netSize*ctx.height.last, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else
                MPI_Recv(&A_h[ctx.netSize*i*ctx.height.basic - ctx.netSize], ctx.netSize*ctx.height.middle, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        showResults(A_h, A, ctx.size, ctx.netSize);
    }
    else{
        cudaMemcpy(A_h, A, sizeof(double)*ctx.size, cudaMemcpyDeviceToHost);
        MPI_Send(A_h, ctx.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    int rank, count;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &count);

    selectDevice(rank, count);

    double accuracy;
    int netSize=0, itCountMax;
    args_parser(argc, argv, accuracy, netSize, itCountMax);
    
    Context ctx = getContext(rank, count, netSize);

    double loss = 0;
    int itCount;

    double *A, *Anew, *Delta, *A_h;
    bufferInitialize(A, Anew, Delta, A_h, ctx);
    
    
    double* val;
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    reductionInitialize(val, Delta, d_temp_storage, temp_storage_bytes, ctx.size);

    

    if (rank!=0)
        cudaDeviceEnablePeerAccess(rank-1,0);
    if (rank!=count-1)
        cudaDeviceEnablePeerAccess(rank+1,0);
    
    dim3 threads(32, 32, 1);
    dim3 blocks(std::ceil((double)netSize/32), std::ceil((double)ctx.height.self/32), 1);


    
    
    bool running = true;
    
    for(itCount = 0; itCount < itCountMax && running; itCount++)
    {
        shareLimits(A, ctx);
        
        solve<<<blocks, threads, 0, ctx.stream>>>(A, Anew, netSize, ctx.height.self);
        catchFailure("solve", ctx.stream);
        if(itCount%100 == 0 || itCount + 1 == itCountMax) { // calc loss every 100 iterations or last
            checkAccuracy(Anew, A, Delta, ctx, accuracy, &loss, &running, d_temp_storage, temp_storage_bytes, val);
        }
        std::swap(A, Anew); // swap pointers on cpu
    }

    catchFailure("calc", ctx.stream);
    if(netSize <= 32){
        getResultsAndShow(A_h, A, ctx);
    }
    
    catchFailure("calc fail", ctx.stream);
    if(rank == 0){
        std::cout << loss << '\n';
        std::cout << itCount << '\n';
    }

    cudaFree(A);
    cudaFree(Anew);
    cudaFree(Delta);
    cudaStreamDestroy(ctx.stream);
    delete[] A_h;
    MPI_Finalize();
    return 0;
}
