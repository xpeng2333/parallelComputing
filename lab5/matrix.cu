#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLK 8
#define dimA (10*BLK*10*BLK)
#define dimB (10*BLK*20*BLK)
#define szA (10*BLK*10*BLK*sizeof(double))
#define szB (10*BLK*20*BLK*sizeof(double))
#define A ((const double (*)[10*BLK])a)
#define B ((const double (*)[20*BLK])b)
#define C ((double (*)[20*BLK])c)
#define bx blockIdx.x
#define by blockIdx.y
#define tx threadIdx.x
#define ty threadIdx.y

void init(int n,double *M){
    int i;
    for(i=0;i<n;i++){
        M[i]=(double)rand()/RAND_MAX;
    }
}
void check(int n,double *x,double *y){
    int i;
    double maxerr=0;
    for(i=0;i<n;i++){
        if(fabsf(x[i]-y[i])/y[i]>maxerr){
            maxerr=fabsf(x[i]-y[i])/y[i];
        }
    }
    printf("max err = %g\n",maxerr);
}
void host_mm(const double *a,const double *b,double *c){
    int i,j,k;
    for(i=0;i<10*BLK;i++){
        for(j=0;j<20*BLK;j++){
            for(k=0;k<10*BLK;k++){
                C[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
}

void print(double *c){
    int i,j;
    for(i=0;i<10*BLK;i++){
        for(j=0;j<20*BLK;j++){
            printf("%.2f\t",C[i][j]);
        }
        printf("\n");
    }
}

__global__
void device_mm(const double *a,const double *b,double *c){
    int k;
    for(k=0;k<10*BLK;k++)
        C[bx*BLK+tx][by*BLK+ty]+=A[bx*BLK+tx][k]*B[k][by*BLK+ty];
}

__global__
void tiled_device_mm(const double *a,const double *b,double *c){
    __shared__ double sA[BLK][BLK];
    __shared__ double sB[BLK][BLK];
    int s,i;
    double sum=0;
    for(s=0;s<10;s++){
        sA[tx][ty]=A[bx*BLK+tx][s*BLK+ty];
        sB[tx][ty]=B[s*BLK+tx][by*BLK+ty];
        __syncthreads();
        for(i=0;i<BLK;i++){
            sum+=sA[tx][i]*sB[i][ty];
        }
        __syncthreads();
    }
    C[bx*BLK+tx][by*BLK+ty]=sum;
}



int main(){
    clock_t start,finish;
    double hosttime,devicetime;

    dim3 grid(BLK,BLK);
    dim3 block(10,20);

    double *hA,*hB,*rC,*dA,*dB,*dC,*hC;
    hA=(double*)malloc(szA);
    hB=(double*)malloc(szB);
    hC=(double*)malloc(szB);
    rC=(double*)malloc(szB);

    init(dimA,hA);
    init(dimB,hB);
    memset(hC,0,szB);

    start=clock();
    host_mm(hA,hB,hC);
    finish=clock();
    hosttime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("cpu time: %.3f\n",hosttime);
    cudaMalloc(&dA,szA);
    cudaMalloc(&dB,szB);
    cudaMalloc(&dC,szB);
    cudaMemset(dC,0,szB);    
    start=clock();
    cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice);
    device_mm<<<block,grid>>>(dA,dB,dC);
    cudaThreadSynchronize();
    cudaMemcpy(rC,dC,szB,cudaMemcpyDeviceToHost);
    finish=clock();    
    devicetime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("gpu time1: %.3f, speedup=%.3f\n",devicetime,hosttime/devicetime);
    cudaMemset(dC,0,szB);    
    start=clock();
    cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice);
    tiled_device_mm<<<block,grid>>>(dA,dB,dC);
    cudaThreadSynchronize();
    cudaMemcpy(rC,dC,szB,cudaMemcpyDeviceToHost);
    finish=clock();    
    devicetime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("gpu time2: %.3f, speedup=%.3f\n",devicetime,hosttime/devicetime);

    free(hA);
    free(hB);
    free(hC);
    free(rC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}