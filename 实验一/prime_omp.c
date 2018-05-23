#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

int isPrime(int n) {
    int i;
    for (i = 2; i <= sqrt(1.0 * n); i++)
        if (n % i == 0)
            return 0;
    return 1;
}
int main() {
    int N_table[4] = {1000, 10000, 100000, 500000};
    int NUM_THREADS_table[4] = {1, 2, 4, 8};
    for (int n = 0; n < 4; n++) {
        for (int tn = 0; tn < 4; tn++) {
            int N = N_table[n];
            int NUM_THREADS = NUM_THREADS_table[tn];
            printf("规模 %d , 线程数 %d  ", N, NUM_THREADS);
            omp_set_num_threads(NUM_THREADS);
            int i, num = 0;
            struct timeval t1, t2, t3, t4;
            double totalTime1, totalTime2;
            gettimeofday(&t1, NULL);
            for (int k = 0; k < 100; k++) {
#pragma omp parallel for reduction(+ : num)
                for (i = 2; i <= N; i++) {
                    num += isPrime(i);
                }
            }
            gettimeofday(&t2, NULL);
            // printf("素数共有 %d 个\n", num);
            totalTime1 =
                (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
            totalTime1 /= 100.0;
            printf("并行时间: %lf us  ", totalTime1);

            num = 0;
            gettimeofday(&t3, NULL);
            for (int k = 0; k < 100; k++) {
                for (i = 2; i <= N; i++) {
                    num += isPrime(i);
                }
            }
            gettimeofday(&t4, NULL);
            totalTime2 =
                (t4.tv_sec - t3.tv_sec) * 1000000 + (t4.tv_usec - t3.tv_usec);
            totalTime2 /= 100.0;
            // printf("素数共有 %d 个\n", num);
            printf("串行时间: %lf us  ", totalTime2);
            printf("加速比: %lf\n", (double)totalTime2 / (double)totalTime1);
        }
        printf("\n");
    }
    return 0;
}