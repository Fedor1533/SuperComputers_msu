#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <fstream>

#include "includes/calculations.h"

using namespace std;
#define M 40
#define N 40
#define delta 1e-7

template <size_t rows, size_t cols>
void save_array_csv(const double (&res)[rows][cols], const string& filename) 
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            file << res[i][j];
            if (j < cols - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

// f(x, y) = 1; D: {(x, y): x^2 + 4y^2 < 1}; 
// w^(0) = 0; delat=10^(-6)

int main(int argc, char *argv[])
{
    // calculate grid cell W, H
    const double h1 = 2.0 / M;
    const double h2 = 1.0 / N;
    // print parameters
    cout << "N: " << N << " M: " << M << endl << "h1: " << h1 << " h2: " << h2 << endl;

    // init array of coords
    Point coords[N + 1][M + 1];
    init_coords_arr(coords, h1, h2);

    // init F_ij
    double F[N + 1][M + 1] = {0};
    init_F_arr(F, coords, h1, h2);
    print_array(F, "F_ij");
    // save_array_csv(F, "F_matrix_40.csv");

    // init a_ij and b_ij
    double a[N + 1][M + 1] = {0};
    double b[N + 1][M + 1] = {0};
    init_coef_arr(a, coords, h1, h2, 'a');
    print_array(a, "Coef_A_ij");
    init_coef_arr(b, coords, h1, h2, 'b');
    print_array(b, "Coef_B_ij");

    // init array of weights W
    double w[N + 1][M + 1] = {0};

    // main cycle to find weights
    double w_new[N + 1][M + 1] = {0};
    double residual[N + 1][M + 1] = {0}; // w_new - w
    double r_k[N + 1][M + 1] = {0};
    double Ar_k[N + 1][M + 1] = {0};
    double tau, Ar_k_norm, res_norm = 0;
    // auto start = chrono::high_resolution_clock::now(); // засекаем время старта
    double start = omp_get_wtime();
    for (int k = 1; ; k++)
    {
        if (k % 100 == 0) { printf("\n------------Step %d: Try to find solution------------\n", k); }
        // get r_k = A*w_k - B
        operator_A(r_k, w, a, b, h1, h2); // A*w_k
        matrix_sub(r_k, r_k, F); // A*w_k - F

        // get tau_k+1 = (A*r_k, r_k) / ||A*r_k||^2
        operator_A(Ar_k, r_k, a, b, h1, h2); // A*r_k
        tau = scalar(Ar_k, r_k, h1, h2); // (A*r_k, r_k)
        Ar_k_norm = scalar(Ar_k, Ar_k, h1, h2); // ||A*r_k||^2
        tau = tau / Ar_k_norm; // tau / ||A*r_k||^2

        // update weights: w_k+1 = w_k - tau*r_k
        matrix_coef(r_k, r_k, tau); // tau * r_k
        matrix_sub(w_new, w, r_k); // w_k - tau * r_k

        // stop if ||w_new - w|| < delta
        matrix_sub(residual, w_new, w); // w_new - w
        res_norm = pow(scalar(residual, residual, h1, h2), 0.5); // ||residual||

        if (k % 100 == 0) { printf("Step %d: Residual norm: %.8f", k, res_norm); }
        if (res_norm < delta) {
            printf("\n*************** Step %d: Found solution ***************\n", k);
            print_array(w_new, "Final weights");
            // save_array_csv(r_k, "w_matrix_40.csv");
            operator_A(r_k, w_new, a, b, h1, h2);
            print_array(r_k, "A*w_new"); 
            // save_array_csv(r_k, "Aw_matrix_40.csv");
            break;
        }
        else {
            matrix_copy(w, w_new); // w = w_new
        }
    }
    //auto stop = chrono::high_resolution_clock::now(); // засекаем время окончания
    double stop = omp_get_wtime();
    // auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    //cout << "\nTime taken by method: " << duration.count() << " milliseconds" << endl;
    printf("\nTime taken by method: %f seconds\n", stop - start);
}