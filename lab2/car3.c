#include <ctype.h>
#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const int MAX_V = 20;
const float P = 0.1;

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
    int max_len = circleNum * MAX_V;
    unsigned int *base_ptr = NULL;
    printf("haha\n");
    car localTable[carPerRank];
    printf("hehe\n");
    for (i = 0; i < carPerRank; i++) {
        localTable[i].v = localTable[i].x = 0;
        localTable[i].d = UINT32_MAX;
        localTable[i].flag_speed = false;
        localTable[i].flag_slow = false;
    }

    if (ProcRank == 0)
        MPI_Win_allocate_shared(max_len * sizeof(MPI_INT), sizeof(MPI_INT),
                                win_info, shmcomm, &base_ptr, &win);
    else
        MPI_Win_allocate_shared(0, sizeof(MPI_INT), win_info, shmcomm,
                                &base_ptr, &win);
    MPI_Info_free(&win_info);

    int dispunit;
    MPI_Aint sz;
    int *my_ptr = NULL;
    int tmp;
    MPI_Win_lock_all(0, win);
    MPI_Win_shared_query(win, 0, &sz, &dispunit, &my_ptr);
    if (ProcRank == 0) {
        my_ptr[0] = carNum;
        for (i = 1; i < max_len; i++) {
            my_ptr[i] = 0;
        }
    }

    srand((unsigned int)time(NULL) + (unsigned int)ProcRank);
    MPI_Barrier(shmcomm);
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
            for (i = localTable[c].x + 1; i < max_len; i++) {
                if (my_ptr[i] > 0)
                    break;
            }
            tmp = i - localTable[c].x;
            if ((i < max_len) && tmp <= localTable[c].v) {
                localTable[c].d = tmp;
                localTable[c].flag_slow = true;
                localTable[c].flag_speed = false;
            }
            if (localTable[c].v > 0 && (rand() * 1.0 / RAND_MAX < P))
                localTable[c].v--;
        }
        MPI_Barrier(shmcomm);
        for (c = 0; c < carPerRank; c++) {
            // printf("%d-%d\n", ProcRank, my_ptr[localTable[c].x]);
            my_ptr[localTable[c].x]--;
            // printf("%d-%d\n", ProcRank, my_ptr[localTable[c].x]);
            localTable[c].x += localTable[c].v;
            my_ptr[localTable[c].x]++;
        }
    }
    MPI_Win_unlock_all(win);
    if (ProcRank == 0) {
        finish = clock();
        int count = 0;
        for (int i = 0; i < max_len; i++) {
            for (int j = 0; j < my_ptr[i]; j++) {
                printf("%d %d\n", count + j, i);
            }
            count += my_ptr[i];
        }
        printf("total time: %lf s\n",
               (double)(finish - start) / CLOCKS_PER_SEC);
    }

    MPI_Win_free(&win);
    MPI_Comm_free(&shmcomm);
    MPI_Finalize();
    return 0;
}
