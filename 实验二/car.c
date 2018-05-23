#include "mpi.h"
#include <ctype.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_V 100
#define P 0.1

typedef struct car {
    unsigned int x; //车位置
    unsigned int v; //车速度
    unsigned int d;
    bool flag_speed;
    bool flag_slow;
} car;

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        printf("参数错误！\n");
        exit(-1);
    }
    int carNum = atoi(argv[1]);
    int circleNum = atoi(argv[2]);
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
    unsigned int *base_ptr = NULL;
    unsigned int LOCAL_MEM_SZ = (carNum + ProcNum - 1) / ProcNum;
    car carTable[LOCAL_MEM_SZ];
    for (int i = 0; i < LOCAL_MEM_SZ; i++) {
        carTable[i].v = carTable[i].x = 0;
        carTable[i].d = UINT32_MAX;
        carTable[i].flag_speed = false;
        carTable[i].flag_slow = false;
    }
    MPI_Win_allocate_shared(LOCAL_MEM_SZ * sizeof(unsigned int),
                            sizeof(unsigned int), win_info, shmcomm, &base_ptr,
                            &win);
    MPI_Info_free(&win_info);

    int dispunit;
    MPI_Aint sz;
    unsigned int *all_ptr[ProcNum];
    unsigned int *my_ptr = NULL;
    unsigned int *localmem =
        (unsigned int *)malloc(LOCAL_MEM_SZ * sizeof(unsigned int));
    if (localmem == NULL) {
        printf("分配空间失败!\n");
        exit(-1);
    }
    for (int i = 0; i < ProcNum; i++) {
        all_ptr[i] = NULL;
    }
    MPI_Win_lock_all(0, win);
    for (int i = 0; i < ProcNum; i++) {
        MPI_Win_shared_query(win, i, &sz, &dispunit, &(all_ptr[i]));
    }

    my_ptr = all_ptr[ProcRank];

    srand((unsigned int)time(NULL) + (unsigned int)ProcRank);
    if (ProcRank == 0)
        start = clock();
    while (circleNum-- > 0) {
        MPI_Barrier(shmcomm);
        for (int c = 0; c < LOCAL_MEM_SZ; c++) {
            if (carTable[c].flag_speed && (carTable[c].v < MAX_V)) {
                carTable[c].v++;
            }
            if (carTable[c].flag_slow) {
                carTable[c].v = carTable[c].d - 1;
            }
            carTable[c].flag_speed = true;
            carTable[c].flag_slow = false;
            for (int i = 0; i < ProcNum; i++) {
                unsigned int *tmpPtr = all_ptr[i];
                for (int j = 0; j < LOCAL_MEM_SZ; j++) {
                    if (tmpPtr[j] > carTable[c].x) {
                        carTable[c].flag_speed = false;
                        if (carTable[c].d > (tmpPtr[j] - carTable[c].x)) {
                            carTable[c].d = (tmpPtr[j] - carTable[c].x);
                            if (carTable[c].d <= carTable[c].v) {
                                carTable[c].flag_slow = true;
                            }
                        }
                    }
                }
            }
            if ((rand() * 1.0 / RAND_MAX < P) && carTable[c].v > 0)
                carTable[c].v--;
        }
        for (int c = 0; c < LOCAL_MEM_SZ; c++) {
            carTable[c].x += carTable[c].v;
        }
        for (int c = 0; c < LOCAL_MEM_SZ; c++) {
            localmem[c] = carTable[c].x;
        }
        MPI_Barrier(shmcomm);
        memcpy(my_ptr, localmem, LOCAL_MEM_SZ * sizeof(unsigned int));
    }

    MPI_Win_unlock_all(win);
    if (ProcRank == 0) {
        finish = clock();
        for (int i = 0; i < ProcNum; i++) {
            unsigned int *tmpPtr = all_ptr[i];
            for (int j = 0; j < LOCAL_MEM_SZ; j++) {
                printf("%d %d\n", i * LOCAL_MEM_SZ + j, tmpPtr[j]);
            }
        }
        printf("total time: %lf s\n",
               (double)(finish - start) / CLOCKS_PER_SEC);
    }

    MPI_Win_free(&win);
    MPI_Comm_free(&shmcomm);
    MPI_Finalize();
    return 0;
}
