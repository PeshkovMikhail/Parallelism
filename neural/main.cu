#include <iostream>
#include <cstdarg>
#include <exception>
#include <string>
#include <cublas_v2.h>

#include "npy.h"

cublasHandle_t handle;

class Shape{
    size_t _ndim;
    size_t* _shape;
public:
    explicit Shape(size_t ndim, ...)
    {
        _shape = new size_t[ndim];
        _ndim = ndim;
        va_list ap;
        va_start(ap, ndim);
        for(int i = 0; i < ndim; i++){
            _shape[i] = va_arg(ap, int);
        }
        va_end(ap);
    }

    Shape(Shape const &shape)
    {
        _ndim = shape._ndim;
        _shape = new size_t[_ndim];
        memcpy(_shape, shape._shape, sizeof(size_t)*_ndim);
    }

    explicit Shape(const std::vector<npy::ndarray_len_t>& shape)
    {
        _ndim = shape.size();
        _shape = new size_t[_ndim];
        for(int i = 0; i < _ndim; i++)
            _shape[i] = shape[i];
    }

    size_t operator[](int i) const{
        if( i < 0 && _ndim + i >= 0)
            return _shape[_ndim+i];
        if(i < _ndim)
            return _shape[i];
        else
            throw std::runtime_error("wrong argument");
    }

    friend bool operator==(const Shape& a, const Shape& b)
    {
        if(a._ndim != b._ndim)
            return false;
        for(int i = 0; i < a._ndim; i++)
        {
            if(a[i] != b[i])
                return false;
        }
        return true;
    }

    friend bool operator!=(const Shape& a, const Shape& b){
        return !(a == b);
    }

    size_t totalLength(){
        size_t res = 1;
        for(int i = 0; i < _ndim; i++)
            res *= _shape[i];
        return res;
    }

    [[nodiscard]] size_t ndim() const {return _ndim;}
};

__global__ void sigmoidKernel(float* x, float* out, size_t size)
{
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= size)
        return;
    out[i] = 1.0f/(1.0f + exp(-x[i]));
}

class Tensor{
    float* _data;
    Shape* _shape;
public:
    Tensor(Shape shape){
        _shape = new Shape(shape);
        _data = nullptr;
        cudaMalloc((void**)&_data, sizeof(float)*shape.totalLength());
        cudaMemset(_data, 0, sizeof(float)*shape.totalLength());
    }
    ~Tensor(){
        cudaFree(_data);
    }

    float* data_ptr() {return _data;}

    void load(const std::string &path){
        std::vector<npy::ndarray_len_t> shape;
        std::vector<float> data;
        npy::LoadArrayFromNumpy<float>(path,shape,data);
        std::reverse(shape.begin(), shape.end());
        if(static_cast<Shape>(*_shape) != Shape(shape))
            throw std::runtime_error("wrong shape");

        cudaMemcpy(_data, data.data(), sizeof(float)*_shape->totalLength(), cudaMemcpyHostToDevice);
    }

    Shape shape() const{return *_shape;}

    friend Tensor operator*(const Tensor& a, const Tensor& b) {
        Tensor res(Shape(2, a.shape()[0], b.shape()[1]));
        if(a.shape()[-1] != b.shape()[0])
            throw std::runtime_error("wrong argument");
        float alpha = 1, beta = 1;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)a.shape()[0], (int)b.shape()[0],(int)a.shape()[1],
                    &alpha, a._data, 1,
                    b._data, (int)b.shape()[0],
                    &beta, res._data, 1);
        return res;
    }

    friend Tensor operator+(const Tensor& a, const Tensor& b) {
        Tensor res(b);
        if(a.shape() != b.shape())
            throw std::runtime_error("wrong argument");
        float alpha = 1;
        cublasSaxpy(handle, (int)a.shape().totalLength(), &alpha, a._data, 1, res._data, 1);
        return res;
    }

    friend Tensor sigmoid(const Tensor& input)
    {
        Tensor res(input.shape());
        unsigned int blocks = std::ceil((float)input.shape().totalLength()/1024.0f);
        sigmoidKernel<<<blocks, 1024>>>(input._data, res._data, input.shape().totalLength());
        return res;
    }

    void reshape(Shape new_shape)
    {
        if(new_shape.totalLength() != _shape->totalLength())
        {
            std::cerr << "old: ";
            for(int i = 0; i < _shape->ndim(); i++)
                std::cerr << (*_shape)[i] << " ";
            std::cerr << std::endl << "new: ";

            for(int i = 0; i < new_shape.ndim(); i++)
                std::cerr << new_shape[i] << " ";
            std::cerr << std::endl;

            throw std::runtime_error("invalid shape");
        }

        *_shape = Shape(new_shape);
    }
};

class Linear{
    Tensor *_weights;
    Tensor *_bias;
public:
    Linear(size_t input_size, size_t output_size){
        _weights = new Tensor(Shape(2, input_size, output_size));
        _bias = new Tensor(Shape(1, output_size));
    }

    Linear(){
        _weights = nullptr;
        _bias = nullptr;
    }
    ~Linear(){
        cudaFree(_weights);
    }

    Tensor forward(Tensor& x){
        Tensor res = x * (*_weights);
        return res + *_bias;
    }

    Tensor operator()(Tensor x){
        return forward(x);
    }

    void load(const std::string& path)
    {
        _weights->load(path + ".npy");
        _bias->load(path+"_bias.npy");
        _bias->reshape(Shape(2, 1, _bias->shape()[0]));
    }
};

class Net{
    Linear _fc1, _fc2, _fc3;
public:
    Net(){
        _fc1 = Linear(1024, 256);
        _fc2 = Linear(256, 16);
        _fc3 = Linear(16, 1);
    }
    ~Net() = default;

    Tensor forward(const Tensor& input){
        Tensor x(input);
        if(x.shape().ndim()!=2)
            x.reshape(Shape(2, 1, x.shape()[0]));
        x = sigmoid(_fc1(input));
        x = sigmoid(_fc2(x));
        return sigmoid(_fc3(x));
    }

    Tensor operator()(const Tensor& input){
        return forward(input);
    }

    void load(const std::string& path)
    {
        _fc1.load(path+"fc1");
        _fc2.load(path+"fc2");
        _fc3.load(path+"fc3");
    }
};

int main() {
    cublasCreate_v2(&handle);

    Tensor input(Shape(1, 1024));
    input.load("../weights/input.npy");

    Net net;
    net.load("../weights/");

    Tensor out = net(input);

    cublasDestroy_v2(handle);

    float t;
    cudaMemcpy(&t, out.data_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
    std::cout.precision(8);
    std::cout << t << std::endl;

    return 0;
}
