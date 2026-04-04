#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    #define DP4A(a, b, c) __dp4a(a, b, c)
#else
    inline __device__ int32_t DP4A(int32_t a, int32_t b, int32_t c) {
        int8_t* pa = (int8_t*)&a;
        int8_t* pb = (int8_t*)&b;
        c += pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
        return c;
    }
#endif

constexpr int TILE_SIZE = 32;

__global__ void dain_int8_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    __shared__ alignas(16) int8_t tileA[TILE_SIZE][TILE_SIZE];
    __shared__ alignas(16) int8_t tileB[TILE_SIZE][TILE_SIZE + 4]; 

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    int32_t accum = 0;

    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        if (row < M && (k_tile * TILE_SIZE + tx) < K)
            tileA[ty][tx] = A[row * K + k_tile * TILE_SIZE + tx];
        else tileA[ty][tx] = 0;

        if (col < N && (k_tile * TILE_SIZE + ty) < K)
            tileB[tx][ty] = B[(k_tile * TILE_SIZE + ty) * N + col];
        else tileB[tx][ty] = 0;

        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i += 4) {
            int32_t a_pack = *reinterpret_cast<const int32_t*>(&tileA[ty][i]);
            int32_t b_pack = *reinterpret_cast<const int32_t*>(&tileB[tx][i]);
            accum = DP4A(a_pack, b_pack, accum);
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = accum;
}

__global__ void int_relu_clamp(const int32_t* input, int8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int32_t val = input[idx];
        output[idx] = static_cast<int8_t>(val < 0 ? 0 : (val > 127 ? 127 : val));
    }
}

// Minimal deterministic hash (non-cryptographic but repeatable)
__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

__global__ void simple_hash(const int32_t* data, int size, uint32_t* out_hash) {
    uint32_t h = 0x6a09e667;
    for(int i=0; i<size; i++) {
        h ^= static_cast<uint32_t>(data[i]);
        h = rotr(h, 7) + 0x9e3779b9;
    }
    if (threadIdx.x == 0) *out_hash = h;
}

int main() {
    const int B=1, S=64, D=128, H=256;
    std::vector<int8_t> w1(D*H, 1), w2(H*D, 1), in(B*S*D, 1);
    
    int8_t *d_in, *d_w1, *d_relu, *d_w2;
    int32_t *d_hid, *d_fin;
    uint32_t *d_hash, h_hash;

    cudaMalloc(&d_in, B*S*D); cudaMalloc(&d_w1, D*H);
    cudaMalloc(&d_hid, B*S*H*4); cudaMalloc(&d_relu, B*S*H);
    cudaMalloc(&d_w2, H*D); cudaMalloc(&d_fin, B*S*D*4);
    cudaMalloc(&d_hash, 4);

    cudaMemcpy(d_in, in.data(), in.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, w1.data(), w1.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2.data(), w2.size(), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dain_int8_gemm<<<dim3((H+31)/32, (B*S+31)/32), block>>>(d_in, d_w1, d_hid, B*S, H, D);
    int_relu_clamp<<<(B*S*H+255)/256, 256>>>(d_hid, d_relu, B*S*H);
    dain_int8_gemm<<<dim3((D+31)/32, (B*S+31)/32), block>>>(d_relu, d_w2, d_fin, B*S, D, H);
    simple_hash<<<1, 32>>>(d_fin, B*S*D, d_hash);

    cudaMemcpy(&h_hash, d_hash, 4, cudaMemcpyDeviceToHost);
    std::cout << "DETERMINISTIC_HASH: 0x" << std::hex << h_hash << std::endl;

    return 0;
}