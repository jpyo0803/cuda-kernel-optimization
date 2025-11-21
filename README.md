## How to run example
```sh
$ cd docker && docker-compose up
```

## CUDA SGEMM Kernel Optimization
CUDA 프로그래밍 모델은 계층적 병렬 실행 구조를 기반으로 한다.  
연산은 크게 **Grid → Block → Thread**의 3계층으로 나누어진다.  
(하드웨어적으로는 이 사이에 Warp 계층이 더 존재하지만, CUDA 코드에는 직접 등장하지 않는다.)

NVIDIA GPU의 메모리 계층을 크게 나누면 다음과 같다.

- **Global Memory(GMEM)**: 우리가 흔히 말하는 VRAM으로, 모든 스레드가 접근할 수 있는 가장 큰 메모리 공간이다. 다만 레이턴시와 대역폭 측면에서는 가장 느린 편이다.
- **Shared Memory(SMEM)**: 같은 Block에 속한 스레드들이 함께 사용하는 on-chip 메모리다. 프로그래머가 직접 인덱스를 계산해 읽고 쓰는 일종의 “소프트웨어 관리 캐시” 역할을 한다.
- **Register**: 각 Thread가 독점적으로 사용하는 가장 빠른 저장 공간으로, 스칼라/로컬 변수들이 주로 여기에 매핑된다. CPU의 캐시처럼 자동으로 관리되는 것이 아니라, 컴파일러가 변수들을 레지스터에 배치한다는 점에서 **프로그램이 논리적으로 직접 사용하는 공간**에 가깝다.

이 외에도 Shared Memory와 공간을 공유하는 **L1 캐시**, 모든 SM이 공유하는 **L2 캐시**가 있다.  
다만 이 글에서는 L1 캐시는 직접 다루지 않고, 동일 계층(on-chip)에서 프로그래머가 제어할 수 있는 Shared Memory(SMEM)를 중심으로 설명한다. L2 캐시는 이후에 따로 다룰 예정이다.

메모리의 접근속도와 용량은 반비례  
속도: Register > SMEM > GMEM  
용량: GMEM > SMEM > Register  

![gpu architecture](./images/gpu_arch.png)

## Three-level of Computation Hierarchy
각 CUDA 커널의 실행은 하나의 Grid로 실행된다. 각 Grid는 다수의 Block들로 이루어져 있고 각 Block은 최대 1024개의 Thread로 구성된다. 

![grid of thread blocks](./images/grid_of_thread_blocks.png)

Block들은 여러 SM에 스케줄될 수 있으며, SM당 여러 Block이 동시에 배치될 수 있다. 같은 Block에 속한 Thread들끼리는 SMEM을 통해 데이터 공유가 가능한 반면 다른 Block에 속한 Thread들끼리는 GMEM을 통해서만 데이터 공유가 가능하다.

## CUDA Kernel Launch Example
아래는 간단한 Kernel 호출 예제이다. 커널 호출 안에서 각 Thread는 자신이 속한 Block 좌표와 Thread 좌표를 출력한다.
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void TestKernel() {
  // 간단한 커널 예제
  printf("Block ID: (%d, %d), Thread ID: (%d, %d)\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main() {
    dim3 grid_dim(2, 2);
    dim3 block_dim(3, 3);

    TestKernel<<<grid_dim, block_dim>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

## SGEMM 소개
해당 튜토리얼에서는 SGEMM (Single Precision General Matrix Multiplication)을 다룬다. SGEMM는 $C=\alpha \cdot A \cdot B + \beta \cdot C$ 를 기본형태로 갖는다. 

여기서 A, B는 두 입력 행렬이고 C는 출력 행렬이다. $\alpha$ 와 $\beta$ 는 단순히 행렬 전체에 곱해지는 scalar값이다. 

## SGEMM의 연산적 특성 분석

이해를 돕기위해 두 A, B 행렬 모두 N by N 행렬이라고 하자, 그럼 SGEMM가 연산되며 드는 연산량과 읽고 써야되는 아래와 같다.

Total FLOPS: $2\cdot N^{3} + 3\cdot N^{2} \approx 2\cdot N^{3}$  
Total data to read: $3 \cdot N^{2} \cdot \text{4 Byte} = 12 \cdot N^{2} \text{ Byte}$  
Total data to store: $4 \cdot N^{2} \text{ Byte}$  

먼저 연산을 위해 드는 총 FLOPS는 행렬곱셈 결과 행렬 C의 각 원소마다 dot product를 수행에 대략적으로 $2 \cdot N^{3}$ 만큼의 연산이 든다(N번의 곱셈 + N-1번의 덧셈). 다음으로 $\alpha \cdot C_{new} + \beta \cdot C_{old}$ 연산을 위해 총 $3 \cdot N^{2}$ 만큼의 연산량이 필요하다.

연산을 위해 읽어들이는 데이터의 크기는 행렬 A, B, C를 각각 적도도어도 한번씩 읽어들여야하기 때문에 $3 \cdot N^{2} \text{ Byte}$ 만큼을 GMEM로부터 읽어와야 한다. 결과 행렬 C를 위해 GMEM에 써야하는 데이터의 크기는 $4 \cdot N^{2} \text{ Byte}$ 이다. 

만약 N을 4096이라고 가정했을때는 아래와 같다.

Total FLOPS: 137 GFLOPS  
Total data to read: 201 MB  
Total data to write: 67 MB  

필자가 현재 사용하는 GPU는 RTX 4060 Ti (8GB)이며 이론적 FP32 연산량 상한은 22 TFLOPS이고 Memory Bandwidth는 288 GB/s이다. 

이론적으로는 137 GFLOPS는 RTX 4060 Ti에서 6.2 ms만에 처리되야한다. 메모리 읽기/쓰기에 드는 시간은 0.93 ms 정도 소요되어야 한다. 

**앞으로 등장할 각 최적화 버전에서는 입력행렬의 크기는 모두 4096 by 4096으로 가정한다.**

## cuBLAS Version (matmul_cublas)
cuBLAS는 NVIDIA GPU의 연산 자원을 활용하여 BLAS(Basic Linear Algebra Subprograms) 함수를 가속화하는 소프트웨어 라이브러리이다.

cuBlas를 사용해 행렬곱셈시 약 13 ms의 시간이 소요되었다. 이는 이론적 상한인 6.2 ms 대비 약 50%정도의 성능까지 도달함을 보여준다. 

## Naive Version (matmul_naive)
Naive버전 행렬곱("matmul_naive.cu")는 CuBLAS의 성능(100% 기준)대비 약 1.3% 정도의 성능밖에 내지 못한다. 실제로 연산에 소요된 시간은 830 ms 정도로 실제 하드웨어가 제공하는 연산능력을 거의 활용하지 못하는 것을 확인 할 수 있다.

## Global Memory Coalescing (matmul_coalescing)
Global memory coalescing 버전은 같은 Warp에 속한 Thread들이 동시에 연속된 메모리를 접근할때 여러개의 개별 메모리 접근 연산을 하나로 합쳐 한번에 수행하는 기능을 활용한다. 이때 Thread들이 접근하는 메모리는 연속적(Consecutive)이어야하고 정렬(Aligned)되어있어야 한다. 대부분의 GPU는 32B, 64B, 128B Burst 메모리 접근을 지원한다. 

만약 32개의 Thread가 4B씩 메모리를 접근하는 상황에서 메모리가 연속적이고 정렬되어있으면 32번의 메모리 연산을 1번에 처리할 수 있다.

아래는 Naive버전의 메모리 접근 방식을 보여준다 (실제로 하나의 Warp에는 32개의 Thread가 있지만 간단하게 표현하기 위해 한 Warp당 8개의 Thread로 설정). 같은 Warp에 속한 Thread들이 메모리를 비연속적이게 접근하고 있음을 확인할 수 있다.
![naive_kernel_mem_access](images/naive_kernel_mem_access.png)
위 다이어그램에서 A 행렬에 대해 한 Warp에 속한 1번 Thread가 Red 영역을 접근, 2번 Thread가 Green 영역을 시간이 지남에 따라 오른쪽으로 이동한다고 했을때 Warp의 Thread들은 항상 행 방향으로 가로질러 메모리를 접근(메모리는 열방향으로 연속적)하기에 Global memory coalescing 기능을 전혀 사용하지 못한다.
![naive_kernel_mem_coalescing](images/Naive_kernel_mem_coalescing.png)

아래 다이어그램은 GMEM coalescing을 했을때 메모리 접근 패턴이다. 
![GMEM_coalescing](images/GMEM_coalescing.png)

GMEM coalescing을 활용하는 방법은 한 Warp에 속한 thread의 번호가 열 방향으로 증가하도록 설정하면된다.

```cpp
int row = blockIdx.x * kBlockDim + (threadIdx.x / kBlockDim);
int col = blockIdx.y * kBlockDim + (threadIdx.x % kBlockDim); // 열방향으로 먼저 증가
```

Memory coalescing 적용후 성능은 cuBLAS 대비 13.5%정도까지 성능이 향상됨을 확인할 수 있었음.

## Shared Memory Cache-Blocking (matmul_smem_block)
GPU의 각 Streaming Multiprocessor (SM)은 L1 캐시와 공간을 공유하는 Shared memory (SMEM)가 존재한다. RTX 4060 Ti (compute capability 8.9)는 각 Block당 사용할 수 있는 최대 shared memory 크기는 약 100 KB이다. SMEM의 대역폭은 일반적으로 GMEM보다 10~20배 빠른 것으로 알려져있다 (Volta architecture 기준). 

행렬 곱셈시 입력 행렬의 각 원소는 결과 계산을 위해 총 N번씩 접근된다. 매번 GMEM을 다시 접근하는 비효율적인 방식대신 데이터를 GMEM에서 SMEM으로 한번 옮겨두고 여러번 재사용하는 방식이 훨씬 더 효율적이다. smem_block 예제에서는 GMEM에서 한번 SMEM으로 데이터를 가져오고 재사용하는 방식으로 GMEM 접근을 최소화하여 성능을 향상시킨다.

![shared_memory_cache_blocking](images/cache-blocking.png)

SMEM cache-blocking방식을 활용해 실제 연산 수행시간을 측정해보았을때 cuBLAS 대비 약 11%정도의 성능이 나왔다. SMEM 방식은 GMEM coalescing 방식보다 높은 성능이 나온다고 알려져있는데 RTX 4060 Ti에서는 N=4069일때 둘이 비슷한 성능이 나왔다(N이 매우 클때는 SMEM 방식이 더 빠른 것 확인)

GMEM대신 SMEM을 사용한다고하더라도 아직 cuBLAS의 성능과는 많이 차이가 난다. 이러한 차이는 Warp가 실행될동안 어떤 상태에 대부분 머물렀는지 확인함으로서 원인 분석이 가능하다. 

![checking_warp_state](images/kernel_3_profiler_warp_stalls.png)
위 그래프에서 Warp는 대부분 **Stall MIO Throttle**상태에 머물렀음을 확인할 수 있다. 이것의 설명은 아래와 같다.

Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as **shared memory instructions**

즉, Warp가 실제 연산을 수행하는데 보내는 시간보다 SMEM 관련 명령어를 처리하는데 더 시간을 소요한다는 것이다. 우리는 SMEM 또한 최소한 접근하며 연산에 필요한 데이터를 최대한 재사용하는 방식을 찾아야한다.

## 1D Blocktiling for Calculating Multiple Results per Thread (matmul_1d_block_tiling)
NVIDIA GPU 하드웨어에는 SMEM보다 더 빠른 저장장치인 Register가 존재한다. 각 Thread는 작업에 필요한만큼 Register를 할당받아 사용할 수 있다. 이때 Thread들끼리는 Register를 공유하지 않는다는 점에 유의한다. 

1D Blocktiling 방식은 각 Thread가 1차원 형태의 다수 결과를 계산한다. 이 방식은 계산에 필요한 데이터를 SMEM으로부터 Register로 가져오고 여러번 재사용하는 방식을 통해 Arithmetic Intensity(데이터를 메모리로부터 한번 가져와서 얼마나 재사용하는지에 대한 정도)를 증가시킨다. 

$ \text{Aritemetic Intensity} = (\text{연산 횟수}) / (\text{메모리 접근 바이트 수}) $

![1d_block_tiling](images/kernel_4_1D_blocktiling.png)

1D block tiling 방식은 cuBLAS 대비 약 20%의 성능을 보인다. 이전보다 약 2배정도의 성능 향상이 있음을 보인다.

아래 그래프를 확인해보면 Stall MIO Throttle이 확연히 준 것을 확인할 수 있다. 
![checking_warp_state_after_1d_tiling](images/Kernel_4_profiler_warp_stalls.png)

## 2D Blocktiling for Calculating Multiple Results per Thread (matmul_2d_block_tiling)

2D block tiling 방식은 1D 버전의 2D 확장이다. 성능은 cuBLAS 대비 50% 성능까지 보인다.

![2d_block_tiling](images/kernel_5_2D_blocktiling.png)

## Vectorize SMEM and GMEM Accesses
이전의 GMEM coalescing이 warp 단위에서 여러 thread의 메모리 접근을 하나의 트랜잭션으로 통합하는 최적화라면, vectorized memory access는 thread 단위에서 연속된 여러 요소(float4, int4 등)를 한 번의 명령으로 Load/Store하는 최적화이다.

```cpp
    float4 tmp = reinterpret_cast<const float4 *>(&A[ira * K + ica * 4])[0];
    
    // 4개의 float을 한번에 로드 
    As[ira + kBM * ica * 4] = tmp.x;
    As[ira + kBM * (ica * 4 + 1)] = tmp.y;
    As[ira + kBM * (ica * 4 + 2)] = tmp.z;
    As[ira + kBM * (ica * 4 + 3)] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[irb * kBN + icb * 4])[0] =
        reinterpret_cast<const float4 *>(&B[irb * N + icb * 4])[0];
```

Vectorized 기법 적용 후 성능은 cuBLAS 대비 약 60%까지 올라옴을 확인했음.

## Warptiling
Warp tiling은 Block tiling과 Thread tiling 사이에 위치한 병렬 계층임. Block tiling과 Thread tiling만으로는 효율적인 GEMM을 만들기 어려움. CUDA 하드웨어는 실제로 Warp 단위(32 Threads)로 실행을 스케줄링하며, warp 단위 최적화를 하지 않으면 다음의 문제가 생김.

- Shared memory bank conflict 해결이 어려움
- Thread tile이 충분히 재사용되지 않아 Register 재사용성이 낮음.
- Warp scheduler를 충분히 활용하지 못해 ILP, latency hiding 부족
- Tensor Core MMA와 구조적 일치가 안됨

따라서 Block tile 내부를 다시 여러 Warp tile로 나누고, 
각 Warp가 자기 tile을 독립적으로 계산하도록 하는 방식이 Warptiling이다.  
이로 인해 Warp 단위 병렬성이 드러나고, shared memory 접근, register reuse, 
스케줄링 효율이 크게 향상된다.

아래는 Warptiling 기법을 사용했을때 어떤식으로 작업이 세분화되어 처리되는지 보여준다.

![warp_tiling](images/kernel_10_warp_tiling.png)

Warptiling 코드는 위 과정을 주석으로 설명했음.

```cpp
namespace {
constexpr int kNumThreads = 128; // block당 스레드 수
constexpr int kNumThreadsPerWarp = 32; // warp당 스레드 수
constexpr int kNumWarps = kNumThreads / kNumThreadsPerWarp; // block당 warp 수

constexpr int kBM = 128; // As의 행 크기 
constexpr int kBN = 128; // Bs의 열 크기
constexpr int kBK = 16;  // As, Bs의 공통 축 크기

constexpr int kWM = 64; // 한 Warp가 처리하는 Warptile의 행 크기
constexpr int kWN = 64; // 한 Warp가 처리하는 Warptile의 열 크기
constexpr int kWNITER = 4; // 한 Warptile에서 Warp-subtile의 열 방향 처리 횟수

constexpr int kTM = 8; // 하나의 Warp-subtile에서 한 Thread가 처리하는 영역의 행 크기
constexpr int kTN = 4; // 하나의 Warp-subtile에서 한 Thread가 처리하는 영역의 열 크기

// 한 Warptile에서 처리하는 Warp-subtile의 행 방향 처리 횟수 
constexpr int kWMITER = (kWM * kWN) / (kNumThreadsPerWarp * kTM * kTN * kWNITER);
constexpr int kWSUBM = kWM / kWMITER; // 한 Warp-subtile의 행 크기
constexpr int kWSUBN = kWN / kWNITER; // 한 Warp-subtile의 열 크기

__global__ void SgemmWarpTiling(int M, int K, int N, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  int br = blockIdx.y;  // Grid내 현재 block row
  int bc = blockIdx.x;  // Grid내 현재 block col

  // 현재 스레드가 속한 warp 인덱스
  int warp_idx = threadIdx.x / kNumThreadsPerWarp;

  // 워프의 행, 열 위치 계산
  int wr = warp_idx / (kBN / kWN); // Block내 warp row
  int wc = warp_idx % (kBN / kWN); // Block내 warp col

  // 워프 내에서의 스레드 인덱스 계산
  int thread_idx_in_warp = threadIdx.x % kNumThreadsPerWarp;

  // Warp subtile 내에서의 스레드 행, 열 위치 계산
  int tr = thread_idx_in_warp / (kWSUBN / kTN);
  int tc = thread_idx_in_warp % (kWSUBN / kTN);

  __shared__ float As[kBM * kBK];
  __shared__ float Bs[kBK * kBN];

  A += br * kBM * K; // A 행렬 행 오프셋을 block 위치로 이동
  B += bc * kBN; // B 행렬 열 오프셋을 block 위치로 이동

  // C 행렬의 행과 열 오프셋을 타겟 Block의 시작 위치로 이동
  C += (br * kBM + wr * kWM) * N + bc * kBN + wc * kWN;

  int ira = threadIdx.x / (kBK / 4);  // As내에서 현재 Thread의 행 위치
  int ica = threadIdx.x % (kBK / 4);  // As내에서 현재 Thread의 열 위치
  int row_stride_a = (kNumThreads * 4) / kBK; // 32

  int irb = threadIdx.x / (kBN / 4);  // Bs내에서 현재 Thread의 행 위치
  int icb = threadIdx.x % (kBN / 4);  // Bs내에서 현재 Thread의 열 위치
  int row_stride_b = kNumThreads / (kBN / 4);

  /*
    각 Thread는 논리적으로 (kWMITER * kTM, kWNITER * kTN) 크기의 결과 영역을 담당하지만,
    실제 C 상에서는 warp 내에서 균등 분산되도록 서로 떨어진 위치의 원소들을 맡게 된다.
  */
  float values[kWMITER * kTM * kWNITER * kTN] = {0.0f};

  /*
    각 Thread는 kWMITER by kWNITER 개의 타일을 순차적으로 처리하게 된다.
    각 타일은 Partial outer product를 계산하기 위해 kWMITER * kTM크기의 A 행렬 조각과
    kWNITER * kTN 크기의 B 행렬 조각을 필요로 한다.
  */ 
  float reg_m[kWMITER * kTM] = {0.0f};
  float reg_n[kWNITER * kTN] = {0.0f};

  for (int bk_off = 0; bk_off < K; bk_off += kBK) {
    // 각 Thread가 협력적으로 자신이 담당하는 As 부분을 GMEM에서 SMEM으로 로드
    for (int offset = 0; offset < kBM; offset += row_stride_a) {
      float4 tmp = reinterpret_cast<const float4 *>(
          &A[(ira + offset) * K + ica * 4])[0];
      As[ira + offset + kBM * ica * 4] = tmp.x;
      As[ira + offset + kBM * (ica * 4 + 1)] = tmp.y;
      As[ira + offset + kBM * (ica * 4 + 2)] = tmp.z;
      As[ira + offset + kBM * (ica * 4 + 3)] = tmp.w;
    }

    // 각 Thread가 협력적으로 자신이 담당하는 Bs 부분을 GMEM에서 SMEM으로 로드
    for (int offset = 0; offset < kBK; offset += row_stride_b) {
      reinterpret_cast<float4 *>(
          &Bs[(irb + offset) * kBN + icb * 4])[0] =
          reinterpret_cast<const float4 *>(&B[(irb + offset) * N + icb * 4])[0];
    }

    // 모든 쓰레드가 데이터 로드를 마칠 때까지 대기
    __syncthreads();

    // 한 Warp가 맡은 Warptile에 대해 kBK 축을 따라 순차적으로 처리
    for (int k = 0; k < kBK; ++k) {
      /* 현재 Thread는 자신이 처리할 warp subtile의 일부분을 register로 로드
         여기서 Tricky한 점은 현재 Thread가 모든 warp subtile에 대해 
         자신이 처리할 여러 부분의 데이터를 레지스터로 한번에 로드한다는 점이다

         예를들어 만약 kWMITER가 2, kWNITER가 2이고 어떤 Thread가 
         (0, 0)위치의 warp subtile에서 (1, 3) 위치의 결과를 담당하면
         (0, 0)위치 subtile 뿐만 아니라 (0, 1), (1, 0), (1, 1) 위치의 subtile에 대해서도
         (1, 3) 위치의 결과를 계산해야 하므로 이들에 대한 A, B 행렬 조각도
         레지스터로 미리 로드해놓아야 한다.
      */
      for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
        for (int i = 0; i < kTM; ++i) {
          reg_m[wsubrow * kTM + i] =
              As[k * kBM + wr * kWM + wsubrow * kWSUBM + tr * kTM + i];
        }
      }
      for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
        for (int j = 0; j < kTN; ++j) {
          reg_n[wsubcol * kTN + j] =
              Bs[k * kBN + wc * kWN + wsubcol * kWSUBN + tc * kTN + j];
        }
      }

      // 한 Thread가 (kWMITER, kWNITER)만큼의 warp subtile을 순차적으로 처리
      for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
        for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
          // 각 Thread는 partial outer product 수행
          for (int i = 0; i < kTM; ++i) {
            for (int j = 0; j < kTN; ++j) {
              values[(wsubrow * kTM + i) * (kWNITER * kTN) + wsubcol * kTN + j] +=
                  reg_m[wsubrow * kTM + i] * reg_n[wsubcol * kTN + j];
            }
          }
        }
      }
    }

    A += kBK;
    B += kBK * N;

    __syncthreads();
  }

  // 결과 행렬 C에 지금까지 계산한 값을 저장
  for (int wsubrow = 0; wsubrow < kWMITER; ++wsubrow) {
    for (int wsubcol = 0; wsubcol < kWNITER; ++wsubcol) {
      float* C_inter = &C[(wsubrow * kWSUBM) * N + wsubcol * kWSUBN];
      for (int i = 0; i < kTM; ++i) {
        for (int j = 0; j < kTN; j += 4) {
          float4 tmp = reinterpret_cast<float4 *>(&C_inter[(tr * kTM + i) * N + tc * kTN + j])[0];

          int idx = (wsubrow * kTM + i) * (kWNITER * kTN) + wsubcol * kTN + j;
          tmp.x = alpha * values[idx] + beta * tmp.x;
          tmp.y = alpha * values[idx + 1] + beta * tmp.y;
          tmp.z = alpha * values[idx + 2] + beta * tmp.z;
          tmp.w = alpha * values[idx + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(&C_inter[(tr * kTM + i) * N + tc * kTN + j])[0] = tmp;
        }
      }
    }
  }
}
}  // namespace
```

Warptiling 기법 적용후 성능은 cuBLAS 대비 ~85%까지 올라옴을 확인함.

## 참고자료 
[1] https://siboehm.com/articles/22/CUDA-MMM  
[2] https://www.arccompute.io/arc-blog/gpu-101-memory-hierarchy  
[3] https://docs.nvidia.com/cuda/cuda-c-programming-guide/  