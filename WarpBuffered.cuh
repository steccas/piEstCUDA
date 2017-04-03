#include <stdint.h>

/////////////////////////////////////////////////////////////////////////////////////
// Public constants

const unsigned WarpBuffered_K=32;
const unsigned WarpBuffered_REG_COUNT=5;
const unsigned WarpBuffered_STATE_WORDS=64;

const uint32_t WarpBuffered_TEST_DATA[WarpBuffered_STATE_WORDS]={
	0x397359cb, 0x80d68b84, 0xae3b8ff3, 0xdc746510, 0x1ab7fcd7, 0x0f023900, 0x204745a1, 0x22d46a53, 0x08971583, 0x4b2c83a3, 0x6aa202ee, 0x9f82c826, 0xd232ff99, 0xa8c4ae86, 0x92e223b7, 0xd4ba48a9, 0x2994ef5b, 0x484004b6, 0xabe07646, 0x5458a1f9, 0x8fadd8bb, 0xfde85fec, 0xa9045f94, 0xc59528f4, 0xce1aacda, 0x2f2b4428, 0xc06c4bd4, 0x47b6f8f3, 0x620d9875, 0xa9d394bd, 0xe59a5e66, 0xf01af52b, 0x01183150, 0xc568c1f0, 0x0292f802, 0xdbabd6fc, 0xa66c3507, 0xd4ca72c5, 0x15ad8b58, 0xa79f06aa, 0x4bc9b1ae, 0xebdd413a, 0xc6a7e681, 0x41a09cc5, 0x26f92596, 0x8b643533, 0xd148ae5f, 0xe70de462, 0x43d5795b, 0xd967709d, 0x38fadda5, 0x5e398144, 0x4f770c47, 0xe4e0da9e, 0xf493d9a0, 0x70b2444a, 0x4e9bba9c, 0x69ca1f93, 0x64f14608, 0xff2d4f12, 0xc66001fa, 0x4caa2945, 0x44c11cfc, 0xb9690718
};

//////////////////////////////////////////////////////////////////////////////////////
// Private constants

const char *WarpBuffered_name="WarpRNG[BufferedU32Rng;k=32;g=16;rs=1;w=32;n=2048;hash=1002e44df84425aa]";
const unsigned WarpBuffered_N=2048;
const unsigned WarpBuffered_W=32;
const unsigned WarpBuffered_G=16;
const unsigned WarpBuffered_SR=1;
__device__ const unsigned WarpBuffered_Q[3][32]={
  {2,26,11,6,12,8,13,31,16,4,25,3,21,23,1,14,17,7,9,20,27,19,30,22,18,5,24,29,15,0,28,10},
  {8,20,31,21,3,25,26,29,17,30,7,0,11,2,22,12,19,18,27,5,10,1,24,16,4,15,23,13,14,28,9,6},
  {31,30,28,24,10,9,27,13,23,22,3,2,1,21,16,20,25,29,5,11,15,17,7,8,26,14,4,18,6,12,19,0}
};
const unsigned WarpBuffered_Z0=19;
__device__ const unsigned WarpBuffered_Z1[32]={
  9,16,12,16,11,16,13,11,13,16,16,13,9,16,14,9,15,9,15,12,14,15,11,9,15,8,9,16,8,9,12,13};
const unsigned WarpBuffered_SHMEM_WORDS=32;
const unsigned WarpBuffered_GMEM_WORDS=32;


////////////////////////////////////////////////////////////////////////////////////////
// Public functions

__device__ void WarpBuffered_LoadState(const unsigned *seed, unsigned *regs, unsigned *shmem)
{
  unsigned offset=threadIdx.x % 32;  unsigned base=threadIdx.x-offset;
  // setup constants
  regs[0]=WarpBuffered_Z1[offset];
  regs[1]=base + WarpBuffered_Q[0][offset];
  regs[2]=base + WarpBuffered_Q[1][offset];
  regs[3]=base + WarpBuffered_Q[2][offset];
  // Setup state
  unsigned stateOff=blockDim.x * blockIdx.x * 2 + threadIdx.x * 2;
  shmem[threadIdx.x]=seed[stateOff];
  regs[4]=seed[stateOff+1];
}

__device__ void WarpBuffered_SaveState(const unsigned *regs, const unsigned *shmem, unsigned *seed)
{
  unsigned stateOff=blockDim.x * blockIdx.x * 2 + threadIdx.x * 2;
  seed[stateOff] = shmem[threadIdx.x];
  seed[stateOff+1]=regs[4];
}

__device__ unsigned WarpBuffered_Generate(unsigned *regs, unsigned *shmem)
{
  #if __DEVICE_EMULATION__
    __syncthreads();
  #endif
  unsigned t0=shmem[regs[1]], t1=shmem[regs[2]];
  unsigned res=(t0<<WarpBuffered_Z0) ^ (t1>>regs[0]) ^ regs[4];
  regs[4] = shmem[regs[3]];
  
  #if __DEVICE_EMULATION__
    __syncthreads();
  #endif
  shmem[threadIdx.x]=res;
  return t0+t1;
};
