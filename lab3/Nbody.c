#include <ctype.h>
#include <malloc.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct pos {
    double x;
    double y;
} pos;

typedef struct force {
    double x;
    double y;
} force;

typedef struct velocity {
    double x;
    double y;
} velocity;

typedef struct acceleration {
    double x;
    double y;
} acceleration;

typedef struct body {
    pos thisPos;
    velocity thisVelocity;
    acceleration thisAcceleration;
} body;

void compute_force(body *thisBody, body *totalPtr, int bodyNum) {
    double distance = 0.0;
    double disX = 0.0;
    double disY = 0.0;
    double Ftmp;
    force Fxy;
    Fxy.x = 0.0;
    Fxy.y = 0.0;
    for (int i = 0; i < bodyNum; i++) {
        disX = totalPtr[i].thisPos.x - thisBody->thisPos.x;
        disY = totalPtr[i].thisPos.y - thisBody->thisPos.y;
        distance = disX * disX + disY * disY;
        if (distance > 4)
            continue;
        Ftmp = 0.0000667 / distance;
        Fxy.x += Ftmp * disX / sqrt(distance);
        Fxy.y += Ftmp * disY / sqrt(distance);
    }
    thisBody->thisAcceleration.x = Fxy.x / 1000;
    thisBody->thisAcceleration.y = Fxy.y / 1000;
}
void compute_velocities(body *thisBody, double delta_t) {
    thisBody->thisVelocity.x += thisBody->thisAcceleration.x * delta_t;
    thisBody->thisVelocity.y += thisBody->thisAcceleration.y * delta_t;
}

void compute_positions(body *thisBody, double delta_t) {
    thisBody->thisPos.x += thisBody->thisVelocity.x * delta_t +
                           thisBody->thisAcceleration.x * delta_t * delta_t / 2;
    thisBody->thisPos.y += thisBody->thisVelocity.y * delta_t +
                           thisBody->thisAcceleration.y * delta_t * delta_t / 2;
}

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        printf("参数错误！\n");
        exit(-1);
    }
    int bodyNum = atoi(argv[1]);
    int moveTime = atoi(argv[2]);
    if (moveTime <= 0 || bodyNum <= 0) {
        printf("小球数和时间数需大于0！\n");
        exit(-1);
    }
    int i, c;
    double currTime = 0.0;
    double frequency=100.0;
    double delta_t = 1.0 / frequency;
    clock_t start, finish;
    int ProcRank, ProcNum;
    MPI_Comm shmcomm;
    MPI_Init(NULL, NULL);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
    MPI_Win win;
    body *base_ptr = NULL;
    unsigned int localBodyNum = bodyNum / ProcNum;
    if (ProcRank == 0)
        MPI_Win_allocate_shared(bodyNum * sizeof(body), sizeof(body), win_info,
                                shmcomm, &base_ptr, &win);
    else
        MPI_Win_allocate_shared(0, sizeof(body), win_info, shmcomm, &base_ptr,
                                &win);
    MPI_Info_free(&win_info);

    int dispunit;
    MPI_Aint sz;
    body *totalPtr = NULL;
    body *my_ptr = NULL;

    MPI_Win_lock_all(0, win);
    MPI_Win_shared_query(win, 0, &sz, &dispunit, &totalPtr);
    my_ptr = totalPtr + ProcRank * localBodyNum;
    if (ProcRank == 0) {
        for (i = 0; i < bodyNum; i++) {
            my_ptr[i].thisAcceleration.x = 0.0;
            my_ptr[i].thisAcceleration.y = 0.0;
            my_ptr[i].thisPos.x = i / sqrt(bodyNum);
            my_ptr[i].thisPos.y = i % (int)sqrt(bodyNum);
            my_ptr[i].thisVelocity.x = 0.0;
            my_ptr[i].thisVelocity.y = 0.0;
        }
        start = clock();
    }
    while (currTime < moveTime) {
        MPI_Barrier(shmcomm);
        for (c = 0; c < localBodyNum; c++) {
            compute_force(my_ptr + c, totalPtr, bodyNum);
            compute_velocities(my_ptr + c, delta_t);
        }
        MPI_Barrier(shmcomm);
        for (c = 0; c < localBodyNum; c++) {
            compute_positions(my_ptr + c, delta_t);
        }
        currTime += delta_t;
    }
    MPI_Win_unlock_all(win);
    if (ProcRank == 0) {
        finish = clock();
        printf("total time: %lf s\n",
               (double)(finish - start) / CLOCKS_PER_SEC);
    }
    MPI_Win_free(&win);
    MPI_Comm_free(&shmcomm);
    MPI_Finalize();
    return 0;
}
