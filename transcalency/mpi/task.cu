#include <iostream>
#include <sstream>
#include <cmath>
#include <cub/cub.cuh>
#include <mpi.h>

using namespace cub;
//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

typedef struct {
    int basic; // высота на которой основываются все остальные путем добавления строк для граничных условий
    int first;  // высота для первого процесса
    int middle; // высота для блоков между первым и последним
    int last; // высота для последнего процесса
    int self; // высота для этого процесса
} Height;

typedef struct {
    int rank; // номер процесса
    int count; // количество процессов
    int netSize; // размер сети
    int size; // размер области вычислений для этого процесса
    Height height; // высоты для этого и других процессов
} Context;

// Проверка на ошибки исполнения
void catchFailure(const std::string& operation_name) {
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << operation_name << " failed " << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
Парсинг значений командной строки: точность(ограничено до 10^-6), размер сети, количество итераций(до 1000000))
*/
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


/*
Реализация пятиточечного алгоритма
A - исходный массив
Anew - массив для записи результата
netSize - ширина сети
heightLimit - высота сети
*/
__global__ void solve(const double *A, double *Anew, int netSize, int heightLimit)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x == 0 || y == 0 || x >= netSize -1 || y >=heightLimit -1)
        return;
    Anew[y*netSize + x] = 0.25 * (A[(y+1)*netSize + x] + A[(y-1)*netSize + x] + A[y*netSize + x + 1] + A[y*netSize + x - 1]);
}


/*
Поиск разности между Anew и A
Delta - массив с результатом (абсолютные значения)
size - количество значений в Delta
*/
__global__ void getDelta(double* Anew, double* A, double* Delta, int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= size)
        return;
    Delta[x] = std::abs(Anew[x] - A[x]);
}


/*
Обмен граничных условий между соседними процессами
*/
void shareLimits(double* A, Context ctx)
{
    if (ctx.rank!=ctx.count -1)
    {
        MPI_Isend(&A[(ctx.height.self-2)*ctx.netSize],ctx.netSize,MPI_DOUBLE,ctx.rank+1,0,MPI_COMM_WORLD);
        MPI_Recv(&A[(ctx.height.self-1)*ctx.netSize],ctx.netSize,MPI_DOUBLE,ctx.rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    if (ctx.rank!=0)
    {
        MPI_Isend(&A[ctx.netSize],ctx.netSize,MPI_DOUBLE,ctx.rank-1,0,MPI_COMM_WORLD);
        MPI_Recv(A,ctx.netSize,MPI_DOUBLE,ctx.rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
}

/*
Anew - массив с новыми значениями после solve
A - старые значения
Delta - массив для хранения разности между Anew и A
ctx - контекст процесса
accuracy - необходимая точность
loss - указатель на переменную с достигнутой точностью
running - указатель на статус исполнения
d_temp_storage - буфер для редукции
temp_storage_bytes - размер буфера для редукции
val - указатель на переменную в видеопамяти. Используется в поиске ошибки
*/
void checkAccuracy(double* Anew, double* A, double* Delta, Context ctx, double accuracy, double* loss, bool* running, void *d_temp_storage, size_t temp_storage_bytes, double* val)
{
    // поиск разности 
    int delta_blocks = ctx.size/1024;
    if(ctx.size%1024!=0)
        delta_blocks+=1;
    getDelta<<<delta_blocks, 1024>>>(Anew, A, Delta, ctx.size);

    //поиск максимального значения ошибки
    CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Delta, val, ctx.size));
    
    cudaMemcpy(loss, val, sizeof(double), cudaMemcpyDeviceToHost); // извлечение значения ошибки

    if(ctx.rank != 0){
        MPI_Send(loss, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // отослать значение ошибки на 0 процесс
        MPI_Recv(running, 1, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // принять статус исполнения
    }
    else{
        double temp_loss = 0;
        for(int i = 1; i < ctx.count; i++) {
            MPI_Recv(&temp_loss, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            *loss = std::max(*loss, temp_loss); // поиск максимального значения ошибки
        }

        if(*loss <= accuracy) // сменить статус исполнения если достигнуто нужное значение ошибки
            *running = false;

        for(int i = 1; i < ctx.count; i++ )
        {
            MPI_Send(running, 1, MPI_C_BOOL, i, 0, MPI_COMM_WORLD); // обмен статусом
        }
    } 
}


/*
Вывод значений сети. Вызывается на 0 процессе.
A_h - массив на хосте
A - массив на видеокарте
netSize - размер сети
*/
void showResults(double* A_h, double* A, int netSize)
{
    for(int y = 0; y < netSize; y++){
        for(int x = 0; x < netSize; x++) {
            std::cout << A_h[y*netSize + x] << " ";
        }
        std::cout << std::endl;
    }
}


/*
выбор видеокарты. Ограничено 4 видеокартами
*/
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

/*
Вычисление высот для структуры Height
*/
void getHeight(Height* height, int rank, int count, int netSize)
{
    height->basic = netSize/count;
    height->middle = height->basic+2;
    height->first = height->middle-1;
    height->last =height->first + netSize%count;

    if(count == 1)
        height->first = netSize;
    height->self = height->middle;
    if(rank == 0)
        height->self = height->first;
    else if(rank == count - 1)
        height->self = height->last;
}

/*
Создание контекста
*/
void getContext(Context* ctx, int rank, int count, int netSize){
    getHeight(&(ctx->height), rank, count, netSize);

    ctx->rank = rank;
    ctx->count = count;
    ctx->netSize = netSize;
    ctx->size = netSize*ctx->height.self;
}

/*
Инициализация буферов для пятиточечного алгоритма.
Выделяет память для массивов и заполняет граничные условия
На 0 процессе и на хосте и на видеокарте хранится полная сеть, на всех остальных выделенный под них участок сети
*/
void bufferInitialize(double*& A, double*& Anew, double*& Delta, double*& A_h, Context ctx)
{
    cudaMalloc((void**)&A, sizeof(double)*ctx.size);
    cudaMalloc((void**)&Anew, sizeof(double)*ctx.size);
    cudaMalloc((void**)&Delta, sizeof(double)*ctx.size);
    catchFailure("alloc");
    cudaMemset(A, 0, sizeof(double)*ctx.size);

    //шаги для линейной интерполяции
    double hor_top_step = (double)(20 - 10) / (double)(ctx.netSize - 1);
    double hor_down_step = (double)(20 - 30) / (double)(ctx.netSize - 1);
    double ver_left_step = (double)(30 - 10) / (double)(ctx.netSize - 1);
    double ver_right_step = (double)(20 - 20) / (double)(ctx.netSize - 1);
    
    if(ctx.rank == 0)
    {
        size_t A_h_size = ctx.netSize * ctx.netSize;
        A_h = new double[A_h_size];
        std::memset(A_h, 0, sizeof(double)*A_h_size);

        //заполнение граничных условий
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
            
            if(i == ctx.count - 1) // пересылка граничных условий для последнего процесса
            {
                MPI_Send(&A_h[ctx.netSize*ctx.height.basic*i - ctx.netSize], ctx.netSize*ctx.height.last, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            else // для всех остальных кроме 0
                MPI_Send(&A_h[ctx.netSize*ctx.height.basic*i - ctx.netSize], ctx.netSize*ctx.height.middle, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);    
        }        
    }
    else{
        // прием и запись участков
        A_h = new double[ctx.size];
        MPI_Recv(A_h, ctx.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaMemcpy(A, A_h, sizeof(double)*ctx.size, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(Anew, A, sizeof(double)*ctx.size, cudaMemcpyDeviceToDevice);
    catchFailure("copy A to Anew");
}

void reductionInitialize(double*& val, double* Delta, void*& d_temp_storage, size_t& temp_storage_bytes, int size)
{
    cudaMalloc((void**)&val, sizeof(double));
    CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Delta, val, size));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
}

/*
Пересылка значений сети на хост и вывод в консоль
*/
void getResultsAndShow(double* A_h, double* A, Context ctx)
{
    if(ctx.rank == 0){
        cudaMemcpy(A_h, A, sizeof(double)*ctx.size, cudaMemcpyDeviceToHost); // получение значений для 0 процесса
        for(int i = 1; i < ctx.count; i++)
        {
            if(i == ctx.count - 1) // получение значений с последнего процесса
                MPI_Recv(&A_h[ctx.netSize*i*ctx.height.basic - ctx.netSize], ctx.netSize*ctx.height.last, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else // со всех остальных
                MPI_Recv(&A_h[ctx.netSize*i*ctx.height.basic - ctx.netSize], ctx.netSize*ctx.height.middle, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        showResults(A_h, A, ctx.netSize); // вывод
    }
    else{
        cudaMemcpy(A_h, A, sizeof(double)*ctx.size, cudaMemcpyDeviceToHost); // пересылка значений на хост
        MPI_Send(A_h, ctx.size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // пересылка 0 потоку
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
    
    Context ctx;
    getContext(&ctx, rank, count, netSize);

    double loss = 0;
    int itCount;

    double *A, *Anew, *Delta, *A_h;
    bufferInitialize(A, Anew, Delta, A_h, ctx);
    
    
    double* val;
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    reductionInitialize(val, Delta, d_temp_storage, temp_storage_bytes, ctx.size);
    
    dim3 threads(32, 32, 1);
    dim3 blocks(std::ceil((double)netSize/32), std::ceil((double)ctx.height.self/32), 1);

    bool running = true;
    for(itCount = 0; itCount < itCountMax && running; itCount++)
    {
        // пересылка граничных условий
        shareLimits(A, ctx);
        
        // пятиточечный алгоритм
        solve<<<blocks, threads>>>(A, Anew, netSize, ctx.height.self);
        cudaDeviceSynchronize();
        if(itCount%100 == 0 || itCount + 1 == itCountMax) { 
            checkAccuracy(Anew, A, Delta, ctx, accuracy, &loss, &running, d_temp_storage, temp_storage_bytes, val);
        }
        std::swap(A, Anew); 
    }

    catchFailure("calc");
    if(netSize <= 32){
        getResultsAndShow(A_h, A, ctx);
    }
    catchFailure("calc fail");
    // вывод ошибки и количества итераций
    if(rank == 0){
        std::cout << loss << '\n';
        std::cout << itCount << '\n';
    }

    // освобождение памяти
    cudaFree(A);
    cudaFree(Anew);
    cudaFree(Delta);
    cudaFree(val);
    delete[] A_h;
    MPI_Finalize();
    return 0;
}
