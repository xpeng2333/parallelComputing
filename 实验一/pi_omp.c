#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

double f(double a) { return (4.0 / (1.0 + a * a)); }

int main(int argc, char const *argv[]) {
    int N_table[4] = {1000, 10000, 50000, 100000};
    int NUM_THREADS_table[4] = {1, 2, 4, 8};
    for (int n = 0; n < 4; n++) {
        for (int tn = 0; tn < 4; tn++) {
            int N = N_table[n];
            int NUM_THREADS = NUM_THREADS_table[tn];
            printf("规模 %d , 线程数 %d  ", N, NUM_THREADS);
            omp_set_num_threads(NUM_THREADS);
            int i;
            double pi, sum = 0.0, h, x;
            struct timeval t1, t2, t3, t4;
            double totalTime1, totalTime2;
            gettimeofday(&t1, NULL);
            for (int k = 0; k < 100; k++) {
                h = 1.0 / N;
#pragma omp parallel for reduction(+ : sum)
                for (i = 1; i <= N; i++) {
                    sum += f(h * ((double)i - 0.5));
                }
                pi = h * sum;
            }
            gettimeofday(&t2, NULL);
            totalTime1 =
                (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
            // printf("%f ", pi);
            totalTime1 /= 100.0;
            printf("并行时间: %lf us  ", totalTime1);
            gettimeofday(&t3, NULL);
            for (int k = 0; k < 100; k++) {
                h = 1.0 / N;
                sum = 0.0;
                for (i = 1; i <= N; i++) {
                    sum += f(h * ((double)i - 0.5));
                }
                pi = h * sum;
            }
            gettimeofday(&t4, NULL);
            totalTime2 =
                (t4.tv_sec - t3.tv_sec) * 1000000 + (t4.tv_usec - t3.tv_usec);
            totalTime2 /= 100.0;
            // printf("%f ", pi);
            printf("串行时间: %lf us  ", totalTime2);
            printf("加速比: %lf\n", (double)totalTime2 / (double)totalTime1);
        }
        printf("\n");
    }
    return 0;
}
