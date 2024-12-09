#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Kernel de redução: cada bloco soma parte do vetor e armazena o resultado em memória compartilhada.
// No final, o primeiro thread do bloco escreve o resultado parcial no vetor de saída.
__global__ void reduceKernel(const float* __restrict__ d_in, float* __restrict__ d_out, size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float mySum = 0.0f;

    // Carrega dados na memória compartilhada, realizando o "unrolling"
    if (i < n) mySum = d_in[i];
    if (i + blockDim.x < n) mySum += d_in[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    // Redução na memória compartilhada
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // O primeiro thread do bloco escreve o resultado parcial
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// Função host para realizar a redução múltipla até restar apenas um valor
float gpuReduceSum(const float* d_in, size_t n) {
    // Definir tamanho do bloco
    int blockSize = 256;  
    int gridSize = (int)((n + blockSize * 2 - 1) / (blockSize * 2));

    // Alocar memória para resultados parciais
    float *d_intermediate, *d_final;
    cudaMalloc(&d_intermediate, sizeof(float) * gridSize);
    cudaMalloc(&d_final, sizeof(float));

    size_t sharedMemSize = blockSize * sizeof(float);
    // Primeira redução
    reduceKernel<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_intermediate, n);

    // Reduzir até sobrar um bloco
    int s = gridSize;
    while (s > 1) {
        int nextGridSize = (s + blockSize * 2 - 1) / (blockSize * 2);
        reduceKernel<<<nextGridSize, blockSize, sharedMemSize>>>(d_intermediate, d_intermediate, s);
        s = nextGridSize;
    }

    // Copiar o resultado final
    cudaMemcpy(d_final, d_intermediate, sizeof(float), cudaMemcpyDeviceToDevice);

    float h_result;
    cudaMemcpy(&h_result, d_final, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_intermediate);
    cudaFree(d_final);

    return h_result;
}

int main() {
    srand((unsigned)time(nullptr));

    size_t n = 100000001; // tamanho do vetor

    // Alocar vetor na CPU
    float* h_vec = (float*)malloc(n * sizeof(float));

    // Preencher vetor com valores aleatórios
    for (size_t i = 0; i < n; i++) {
        h_vec[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Alocar e copiar vetor para GPU
    float* d_vec;
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMemcpy(d_vec, h_vec, n * sizeof(float), cudaMemcpyHostToDevice);

    // Medição do tempo de execução
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Realizar soma na GPU
    float soma = gpuReduceSum(d_vec, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular o tempo de execução
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Imprime resultado
    std::cout << "Soma do vetor: " << soma << std::endl;
    std::cout << "Tempo de execução: " << milliseconds << " ms" << std::endl;

    // Liberar memória
    cudaFree(d_vec);
    free(h_vec);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}