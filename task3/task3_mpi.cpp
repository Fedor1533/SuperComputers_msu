#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <mpi.h>

#include "includes/calculations.h"

using namespace std;
#define M 80
#define N 80
#define NUM_PROCS 4
#define delta 2.5e-8

// f(x, y) = 1; D: {(x, y): x^2 + 4y^2 < 1}; 
// w^(0) = 0; delat=10^(-6)

int main(int argc, char *argv[])
{
    // calculate grid cell W, H
    const double h1 = 2.0 / M;
    const double h2 = 1.0 / N;

    // init MPI processes
    MPI_Init(&argc, &argv);
    int numprocs, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) { cout << "N: " << N << " M: " << M << endl << "h1: " << h1 << " h2: " << h2 << endl; }
    
    // make grid
    const int divisor = pow(2, floor(log2(NUM_PROCS)/2));
    const int grid_rows = (NUM_PROCS > 1) ? divisor : 1; 
    const int grid_cols = (NUM_PROCS > 1) ? NUM_PROCS / divisor : 1;
    if (my_rank == 0) {std::cout << "Grid: ( "<< grid_rows << ", "  << grid_cols << " )" << std::endl; }
    const int N_domain = (grid_rows > 1) ? N / grid_rows + 2: N + 1; // domain rows
    const int M_domain = (grid_cols > 1) ? M / grid_cols + 2: M + 1; // domain cols
    if (my_rank == 0)  {std::cout << "Process "<< my_rank << " rows: "  << N_domain << " cols: " << M_domain << std::endl; }
    int grid[2] = {grid_rows, grid_cols};
    // Определяем ранги соседних процессов
    std::vector<int> neighbors_rank(4, 0); // top_rank, bottom_rank, left_rank, right_rank
    getNeighborsRank(neighbors_rank, grid, my_rank, numprocs);

    // init local array of coords
    Point coords[N_domain][M_domain];
    init_coords_arr(coords, h1, h2, N_domain, M_domain, grid, my_rank);

    // init F_ij
    double F[N_domain][M_domain] = {0};
    init_F_arr(F, coords, h1, h2);
    printDomains(F, my_rank, numprocs, "F_ij");

    // init a_ij and b_ij
    double a[N_domain][M_domain] = {0};
    double b[N_domain][M_domain] = {0};
    init_coef_arr(a, coords, h1, h2, 'a');
    printDomains(a, my_rank, numprocs, "A_ij");
    init_coef_arr(b, coords, h1, h2, 'b');
    printDomains(b, my_rank, numprocs, "B_ij");

    // init array of weights W
    double w[N_domain][M_domain] = {0};

    // main cycle to find weights
    double w_new[N_domain][M_domain] = {0};
    double residual[N_domain][M_domain] = {0}; // w_new - w
    double r_k[N_domain][M_domain] = {0};
    double Ar_k[N_domain][M_domain] = {0};
    double tau, Ar_k_norm, res_norm = 0;
    double glob_tau, glob_Ar_k_norm, glob_res_norm = 0;
    // measure start time
    double start = omp_get_wtime();
    for (int k = 1; ; k++)
    {
        if (k % 500 == 0 & my_rank == 0) printf("\n------------Step %d: Try to find solution------------\n", k);

        // Обмен граничными данными
        if (numprocs > 1) { exchangeBorders(w, neighbors_rank); }

        // get r_k = A*w_k - B
        operator_A(r_k, w, a, b, h1, h2); // A*w_k
        matrix_sub(r_k, r_k, F); // A*w_k - F

        // Обмен граничными данными
        if (numprocs > 1) { exchangeBorders(r_k, neighbors_rank); }

        // get tau_k+1 = (A*r_k, r_k) / ||A*r_k||^2
        operator_A(Ar_k, r_k, a, b, h1, h2); // A*r_k
        tau = scalar(Ar_k, r_k, h1, h2, grid, my_rank); // local (A*r_k, r_k)
        Ar_k_norm = scalar(Ar_k, Ar_k, h1, h2, grid, my_rank); // local ||A*r_k||^2

        // sum all local taus and all local Ar_k_norms
        MPI_Allreduce(&tau, &glob_tau, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&Ar_k_norm, &glob_Ar_k_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        glob_tau = glob_tau / glob_Ar_k_norm; // get global tau / ||A*r_k||^2

        // update weights: w_k+1 = w_k - tau*r_k
        matrix_coef(r_k, r_k, glob_tau); // tau * r_k
        matrix_sub(w_new, w, r_k); // w_k - tau * r_k

        // residual = ||w_new - w|| 
        matrix_sub(residual, w_new, w); // w_new - w
        res_norm = scalar(residual, residual, h1, h2, grid, my_rank); // get local residual
        // sum all local residuals 
        MPI_Allreduce(&res_norm, &glob_res_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        glob_res_norm = pow(glob_res_norm, 0.5); // get global ||residual||
        if (k % 500 == 0 & my_rank == 0) printf("Step %d: Residual norm: %.10f\n", k, glob_res_norm);
        // stop if ||w_new - w|| < delta
        if (glob_res_norm < delta) {
            if (my_rank == 0) printf("\n*************** Step %d: Found solution ***************\n", k);
            break;
        }
        matrix_copy(w, w_new); // w = w_new
    }
    // Синхронизация
    MPI_Barrier(MPI_COMM_WORLD);
    // Записываем время выполнения вычислений
    double stop = omp_get_wtime();
    if (my_rank == 0) { printf("\nTime taken by method: %f seconds\n", stop - start); }
    // Выведем полученный значения функции и результат применение оператора
    printDomains(w_new, my_rank, numprocs, "Final weights");
    // Обмен граничными данными
    if (numprocs > 1) { exchangeBorders(w_new, neighbors_rank); }
    // Посчитаем A*w_new
    operator_A(r_k, w_new, a, b, h1, h2);
    printDomains(r_k, my_rank, numprocs, "A*w_final");
    // Очищение всех состояний, связанных с MPI
    MPI_Finalize();
    return 0;
}
