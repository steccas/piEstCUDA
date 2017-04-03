/*
	Progetto GPGPU
	Luca Steccanella
	W83_000009

	Monte Carlo Hit or Miss
	Generazione numeri casuali
*/

//Includes

#include "WarpBuffered.cuh" //Libreria MWC64X, ci sono tre implementazioni diverse del warp

#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <climits>
#include <vector>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <numeric> 

//cuda incs

#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#define _CRT_RAND_S
#define restrict __restrict__

using namespace std;

#define BLOCK_SIZE 1024

extern __shared__ unsigned rngShmem[];

__global__ void init(unsigned int seed, curandState_t *states)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    /*if (i >= t)
        return;*/
    curand_init(seed, i, 0, states + i); //funzione proprietaria della curand: genera i semi
} //Kernel di inizializzazione: qui vengono generati i semi per curand

__global__ void mc_curand(int *hits, curandState_t *states, int c)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    /*if (i >= t)
        return;*/

    int tHits = 0;
    
    for (int j=0; j<c; j++){
        //tramite la funzione curand_uniform ottengo un valore casuale

        float x = curand_uniform(states + i);
        float y = curand_uniform(states + i);

        //effettuo il passo dell'algoritmo mc_curand dove si verifica la somma dei quadrati

        float z = (x * x + y * y);
        if (z <= 1.0) tHits++;
    }
    //printf("%d %f %f %f\n", i, x, y, z);
    //printf("tHits: %d ; thread: %d\n", tHits, i);

    hits[i] = tHits;
} //Kernel che esegue l'algoritmo Monte Carlo Hit or Miss usando la curand

__global__ void mc_warp(int *hits, unsigned *state, int c)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    /*if (i >= c)
        return;*/

    //inizializzo l'RNG
    unsigned rngRegs[WarpBuffered_REG_COUNT];
    WarpBuffered_LoadState(state, rngRegs, rngShmem);

    //Ottengo i valori casuali ed effettuo i passi dell'algoritmo come suggerito nella documentazione

    int tHits = 0;

    for(int j=0; j<c; j++){
        unsigned long x = WarpBuffered_Generate(rngRegs, rngShmem);
        unsigned long y = WarpBuffered_Generate(rngRegs, rngShmem);

        x = (x * x) >> 3;
        y = (y * y) >> 3;

        if (x + y <= (1UL << 61))
        {
            tHits++;
        }
    }

    //printf("tHits: %d ; thread: %d\n", tHits, i);

    hits[i] = tHits;

    WarpBuffered_SaveState(rngRegs, rngShmem, state);
} //Kernel che esegua l'algoritmo Monte Carlo Hit or Miss usando MWC64X

void check_error(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s : errore %d (%s)\n",
                msg, err, cudaGetErrorString(err));
        exit(err);
    }
} //controllo errori

__global__ void red(int *vrid, const int *v1, int numels)
{
    __shared__ int local[BLOCK_SIZE];
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int val = 0;

    while (i < numels)
    {
        val += v1[i];
        i += blockDim.x * gridDim.x;
    }

    local[threadIdx.x] = val;

    for (int numthreads = BLOCK_SIZE / 2; numthreads > 0; numthreads /= 2)
    {
        __syncthreads();
        if (threadIdx.x < numthreads)
        {
            local[threadIdx.x] += local[numthreads + threadIdx.x];
        }
    }
    if (threadIdx.x == 0)
        vrid[blockIdx.x] = local[0];
} //kernel che effettua la riduzione, come visto a lezione

cudaEvent_t evt[2]; //array di eventi ad uso generale

void do_rid(int *vrid, const int *v1, int N, int numBlocks)
{
    cudaEventRecord(evt[0]);
    red<<<numBlocks, BLOCK_SIZE>>>(vrid, v1, N);
    cudaEventRecord(evt[1]);

    cudaError_t error = cudaEventSynchronize(evt[1]);
    check_error(error, "rid sync");
} //kernel che gesisce la riduzione

int main(int argc, char **argv)
{
    //controlli avvio

    if (argc != 3)
    {
        fprintf(stderr, "Inserire numero di campioni e il metodo di randomizazione; mcCUDA [campioni] [metodo]\n0: curand\n1: MWC64X (Warp Standard)\n");
        return 1;
    }
    int N = atoi(argv[1]);
    int t = atoi(argv[2]);

    /*if (N >= 2097152 && t == 0)
    {
        fprintf(stderr, "Troppi campioni\n");
        return 1;
    }*/
    if (t < 0 || t > 1)
    {
        fprintf(stderr, "Metodo non valido\n");
        return 1;
    }

    //allocazione di eventi, array hits, etc...

    cudaError_t errore;

    int devId=-1;
    cudaDeviceProp devProps;

    cudaGetDevice(&devId);
    cudaGetDeviceProperties(&devProps, devId);
    unsigned gridSize=devProps.multiProcessorCount;
  
    unsigned totalThreads=BLOCK_SIZE*gridSize;

    int totThrds = (int)totalThreads;
    if(N < totThrds) fprintf(stderr, "Usa almeno: %d campioni\n", totThrds);

    int *d_hits;
    size_t numbytes = totThrds * sizeof(int);

    errore = cudaMalloc((void **)&d_hits, numbytes);
    check_error(errore, "alloc hits");
    errore = cudaMemset(d_hits, 0, numbytes);
    check_error(errore, "memset hits");

    cudaEvent_t prima_init, prima_mc;
    cudaEvent_t dopo_init, dopo_mc;

    errore = cudaEventCreate(&prima_init);
    check_error(errore, "create prima init");
    errore = cudaEventCreate(&prima_mc);
    check_error(errore, "create prima mc");
    errore = cudaEventCreate(&dopo_init);
    check_error(errore, "create dopo init");
    errore = cudaEventCreate(&dopo_mc);
    check_error(errore, "create dopo mc");

    float runtime;
    size_t iobytes;

    int numBlocks = devProps.multiProcessorCount * 6;
    
    int Nt = N/totThrds;
    printf("Nt: %d Threads: %d\n", Nt, totThrds);

    if (t == 1)
    {
        //printf("Prima init\n");

        /*
			Di seguito inizializzo le variabili necessarie al funzionamento della libreria
			e genero i semi che verranno usati dall'RNG; MWC64X non genera i semi nella GPU come CuRAND
	    */

        void *seedDevice = 0;
        unsigned totalRngs = totalThreads / WarpBuffered_K;
        unsigned rngsPerBlock = BLOCK_SIZE / WarpBuffered_K;
        unsigned sharedMemBytesPerBlock = rngsPerBlock * WarpBuffered_K * 4;
        unsigned seedBytes = totalRngs * 4 * WarpBuffered_STATE_WORDS;

        std::vector<uint32_t> seedHost(seedBytes / 4);

        cudaEventRecord(prima_init);
        if (cudaMalloc(&seedDevice, seedBytes))
        {
            fprintf(stderr, "Error couldn't allocate state array of size %u\n", seedBytes);
            exit(1);
        }

        int fr = open("/dev/urandom", O_RDONLY);
        if (seedBytes != read(fr, &seedHost[0], seedBytes))
        {
            fprintf(stderr, "Couldn't seed RNGs.\n");
            exit(1);
        }

        //cudaMemcpy(seedDevice, &seedHost[0], seedBytes, cudaMemcpyHostToDevice);
        cudaEventRecord(dopo_init);

        //printf("Dopo init / prima mc\n");

        //eseguo il kernel

        cudaEventRecord(prima_mc);
        mc_warp<<<gridSize, BLOCK_SIZE, sharedMemBytesPerBlock>>>(d_hits, (unsigned *)seedDevice, Nt);
        //printf("sync2\n");
        errore = cudaDeviceSynchronize();
        check_error(errore, "sync2");
        cudaEventRecord(dopo_mc);
    } //MWC64X

    if (t == 0)
    {
        curandState_t *states;
        errore = cudaMalloc((void **)&states, totThrds * sizeof(curandState_t));
        check_error(errore, "alloc states");

        //printf("Prima init\n");

        cudaEventRecord(prima_init);
        init<<<gridSize, BLOCK_SIZE>>>(time(0), states); //genero i semi nella GPU
        cudaEventRecord(dopo_init);

        //printf("sync1\n");
        errore = cudaDeviceSynchronize();
        check_error(errore, "sync1");

        //printf("Dopo init / prima mc\n");

        cudaEventRecord(prima_mc);
        mc_curand<<<gridSize, BLOCK_SIZE>>>(d_hits, states, Nt); //eseguo il kernel

        //printf("sync2\n");

        errore = cudaDeviceSynchronize();
        check_error(errore, "sync2");
        cudaEventRecord(dopo_mc);
    } //curand

    //printf("Dopo mc / prima cpy\n");

    //numbytes = N*sizeof(int);

    float totRuntime = 0.0;

    errore = cudaEventSynchronize(dopo_mc);
    check_error(errore, "fine di tutto");

    errore = cudaEventElapsedTime(&runtime, prima_init, dopo_init);
    check_error(errore, "time init");
    printf("init %gms\n", runtime);

    totRuntime += runtime;

    errore = cudaEventElapsedTime(&runtime, prima_mc, dopo_mc);
    check_error(errore, "time mc");
    iobytes = totThrds * Nt * sizeof(int);
    printf("mc %gms, %g GB/s\n", runtime, iobytes / (runtime * 1.0e6));

    totRuntime += runtime;

    /*cudaDeviceProp props;

    errore = cudaGetDeviceProperties(&props, 0);

    numBlocks = props.multiProcessorCount * 6;*/

    //effettuo la riduzione (chiamando l'apposito kernel) dell'array delle hits in modo di ottenerne il totale

    int *dvrid = NULL;
    int h_hits_sum = 0;

    size_t ridsize = (numBlocks + 1) * sizeof(int);

    errore = cudaEventCreate(evt);
    check_error(errore, "evt create 0");
    errore = cudaEventCreate(evt + 1);
    check_error(errore, "evt create 1");

    errore = cudaMalloc(&dvrid, ridsize);
    check_error(errore, "malloc dvsum");
    errore = cudaMemset(dvrid, -1, ridsize);
    check_error(errore, "memset dvsum");

    cudaEventRecord(evt[0]);
    do_rid(dvrid + 1, d_hits, totThrds, numBlocks);
    do_rid(dvrid, dvrid + 1, numBlocks, 1);
    cudaEventRecord(evt[1]);
    cudaEventSynchronize(evt[1]);
    errore = cudaEventSynchronize(evt[1]);
    check_error(errore, "rid");
    errore = cudaEventElapsedTime(&runtime, evt[0], evt[1]);
    check_error(errore, "time rid");
    iobytes = totThrds * sizeof(int) * numBlocks;
    printf("rid %gms, %g GB/s\n", runtime, iobytes / (runtime * 1.0e6));

    totRuntime += runtime;

    cudaEventRecord(evt[0]);
    cudaMemcpy(&h_hits_sum, dvrid, sizeof(h_hits_sum), cudaMemcpyDeviceToHost);
    cudaEventRecord(evt[1]);
    errore = cudaEventSynchronize(evt[1]);

    check_error(errore, "memcpy D2H");

    cudaEventDestroy(evt[0]);
    cudaEventDestroy(evt[1]);
    cudaFree(dvrid);

    printf("Total: %gms\n", totRuntime);

    //effettuo il passo finale dell'algoritmo e stampo i risultati
    /*std::vector<uint32_t>hitsHost(totalThreads, 0);
    cudaMemcpy(&hitsHost[0], d_hits, 4*totalThreads, cudaMemcpyDeviceToHost);
    int totalHits = std::accumulate(hitsHost.begin(), hitsHost.end(), 0.0);*/

    cout << "Hits: " << h_hits_sum << "\n";
    //cout << "HitsT: " << totalHits << "\n";

    float AS = float(h_hits_sum) / (totThrds*float(Nt));

    float estPi = AS * 4;

    cout << "Est PI: " << estPi << "\n";
	
	float rError = fabs(1.0 - estPi/M_PI);

	cout << "Relative error: " << rError << "\n";

    //cudaFree(states);
    cudaFree(d_hits);

    return 0;
}