#include "mpi.h"
#include <math.h>
#include <stdio.h>

double f(double a) { return (4.0 / (1.0 + a * a)); }

int main(int argc, char *argv[]) {
    int ProcRank, ProcNum, n = 0, i;
    int N_table[4] = {1000, 10000, 50000, 100000};
    double mypi, pi, h, sum, x, t1, t2, t3, t4, totaltime1, totaltime2;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    for (int k = 0; k < 4; k++) {
        if (ProcRank == 0) {
            n = N_table[k];
            t1 = MPI_Wtime();
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        h = 1.0 / n;
        sum = 0.0;
        for (i = ProcRank + 1; i <= n; i += ProcNum) {
            x = h * ((double)i - 0.5);
            sum += f(x);
        }
        mypi = h * sum;
        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (ProcRank == 0) {
            t2 = MPI_Wtime();
            // printf("pi is approximately %.16f, Error is %.16f\n", pi,
            //       fabs(pi - PI25DT));
            totaltime1 = t2 - t1;
            printf("规模 %d  ", n);
            printf("并行时间 %f  ", totaltime1);
        }
        if (ProcRank == 0) {
            t3 = MPI_Wtime();
            h = 1.0 / n;
            sum = 0.0;
            for (i = 1; i <= n; i++) {
                x = h * ((double)i - 0.5);
                sum += f(x);
            }
            pi = h * sum;
            t4 = MPI_Wtime();
            totaltime2 = t4 - t3;
            // printf("%f ", pi);
            printf("串行时间 %f  ", totaltime2);
            printf("加速比 %f \n", totaltime2 / totaltime1);
        }
    }
    MPI_Finalize();
    return 0;
}