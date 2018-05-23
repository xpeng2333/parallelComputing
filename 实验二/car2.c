#include <ctype.h>
#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const int MAX_V = 100;
const float P = 0.1;

typedef struct car {
    unsigned int x; //车位置
    unsigned int v; //车速度
    unsigned int d;
    bool flag_speed;
    bool flag_slow;
} car;
int cmpCar(const void *a, const void *b) {
    return (((car *)a)->x) - (((car *)b)->x);
}
int main(int argc, char const *argv[]) {
    if (argc != 3) {
        printf("参数错误！\n");
        exit(-1);
    }
    int i, c;
    int carNum = atoi(argv[1]);
    int circleNum = atoi(argv[2]) + 1;
    clock_t start, finish;
    int ProcRank, ProcNum, namelen;
    MPI_Comm shmcomm;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(NULL, NULL);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Get_processor_name(name, &namelen);
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
    MPI_Win win;
    carNum += carNum % ProcNum;
    int carPerRank = carNum / ProcNum;
    unsigned int *base_ptr = NULL;

    if (ProcRank == 0)
        MPI_Win_allocate_shared(carNum * sizeof(car), sizeof(car), win_info,
                                shmcomm, &base_ptr, &win);
    else
        MPI_Win_allocate_shared(0, sizeof(car), win_info, shmcomm, &base_ptr,
                                &win);
    MPI_Info_free(&win_info);
    int dispunit;
    MPI_Aint sz;
    car *my_ptr = NULL;

    MPI_Win_lock_all(0, win);
    MPI_Win_shared_query(win, 0, &sz, &dispunit, &my_ptr);
    int offset = carPerRank * ProcRank;
    my_ptr += offset;
    car localTable[carNum - offset];
    for (i = 0; i < carNum - offset; i++) {
        localTable[i].v = localTable[i].x = 0;
        localTable[i].d = UINT32_MAX;
        localTable[i].flag_speed = false;
        localTable[i].flag_slow = false;
    }
    srand((unsigned int)time(NULL) + (unsigned int)ProcRank);
    if (ProcRank == 0)
        start = clock();

    while (circleNum-- > 0) {
        MPI_Barrier(shmcomm);
        for (c = 0; c < carPerRank; c++) {
            if (localTable[c].flag_speed && (localTable[c].v < MAX_V)) {
                localTable[c].v++;
            } else if (localTable[c].flag_slow) {
                localTable[c].v = localTable[c].d - 1;
                localTable[c].flag_slow = false;
            }
            localTable[c].flag_speed = true;
            for (i = c + 1; i < carNum - offset; i++) {
                if (localTable[i].x > localTable[c].x)
                    break;
            }
            if ((i < carNum - offset) && (localTable[c].d <= localTable[c].v)) {
                localTable[c].flag_slow = true;
                localTable[c].flag_speed = false;
            }

            if (localTable[c].v > 0 && (rand() * 1.0 / RAND_MAX < P))
                localTable[c].v--;
            localTable[c].x += localTable[c].v;
        }
        memcpy(my_ptr, localTable, carPerRank * sizeof(car));
        MPI_Barrier(shmcomm);
        if (ProcRank == 0)
            qsort(my_ptr, carNum, sizeof(car), cmpCar);
        MPI_Barrier(shmcomm);
        memcpy(localTable, my_ptr, (carNum - offset) * sizeof(car));
    }
    MPI_Win_unlock_all(win);
    if (ProcRank == 0) {
        finish = clock();
        for (int i = 0; i < carNum; i++) {
            printf("%d %d\n", i, my_ptr[i].x);
        }
        printf("total time: %lf s\n",
               (double)(finish - start) / CLOCKS_PER_SEC);
    }

    MPI_Win_free(&win);
    MPI_Comm_free(&shmcomm);
    MPI_Finalize();
    return 0;
}
