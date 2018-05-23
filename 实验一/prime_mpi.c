#include <math.h>
#include <mpi.h>
#include <stdio.h>

int isPrime(int n) {
    int i;
    for (i = 2; i <= sqrt(1.0 * n); i++)
        if (n % i == 0)
            return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    int n = 0, myid, numprocs = 4, i, pi, sum = 0, mypi;
    double startwtime1, endwtime1, startwtime2, endwtime2, totaltime1,
        totaltime2;
    int N_table[4] = {1000, 10000, 100000, 500000};
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    for (int k = 0; k < 4; k++) {
        if (myid == 0) {
            n = N_table[k];
            startwtime1 = MPI_Wtime();
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (i = myid * 2 + 1; i <= n; i += numprocs * 2)
            sum += isPrime(i);
        mypi = sum;
        MPI_Reduce(&mypi, &pi, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myid == 0) {
            // printf("结果=%d\n", pi);
            endwtime1 = MPI_Wtime();
            totaltime1 = endwtime1 - startwtime1;
            printf("规模 %d  ", n);
            printf("并行时间 %lf s  ", totaltime1);
        }
        //串行程序
        if (myid == 0) {
            sum = 0;
            startwtime2 = MPI_Wtime();
            for (i = 1; i <= n; i += 2)
                sum += isPrime(i);
            endwtime2 = MPI_Wtime();
            totaltime2 = endwtime2 - startwtime2;
            // printf("结果=%d\n", sum);
            printf("串行时间 %lf s  ", totaltime2);
            printf("加速比 %lf \n", totaltime2 / totaltime1);
        }
    }
    MPI_Finalize();
    return 0;
}