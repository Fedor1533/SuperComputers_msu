#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <vector>
#include<cmath>
#include <iomanip>
#include <omp.h>

using namespace std;

struct Point{ 
   double x;
   double y;
   Point() {}
   Point(double x_, double y_){
    this->x = x_;
    this->y = y_;
   }
};

// operator== for struct Point
bool operator== (const Point& p1, const Point& p2);

// operator!= for struct Point
bool operator!= (const Point& p1, const Point& p2);

// operator << for struct Point
std::ostream& operator<<(ostream &os, Point &p);

// find length of vector
double length(const Point& p1, const Point& p2);

// init 2d array of grid coords
template <size_t rows, size_t cols>
void init_coords_arr(Point (&arr)[rows][cols], const double h1, const double h2)
{

    cout << "\n-------------Init array with coords-------------" << endl;
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            arr[i][j] = Point(-1. + j*h1, 0.5 - i*h2);
}

// print 2d array
template <size_t rows, size_t cols, typename T>
void print_array(T (&arr)[rows][cols], const string name="Matrix")
{
    printf("\n-------------Print array: %s-------------\n", name.c_str());
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
            cout << setprecision(6) << arr[i][j] << string(5, ' '); // setprecision(3)
        cout << endl;
    }
}

// get scalar product for 2d matrixes
template <size_t rows, size_t cols>
double scalar(double (&u)[rows][cols], double (&v)[rows][cols], const double h1, const double h2)
{
    double product = 0;
    #pragma omp parallel for reduction (+:product)
    for (size_t i = 1; i < rows - 1; i++)
        for (size_t j = 1; j < cols - 1; j++) {
            product += h1 * h2 * u[i][j] * v[i][j];
        }

    return product;
}

// get area of polygon(vector of 2d points)
double polygon_area(vector<Point>& polygon);

//check if Point is in the area D: x^2 + 4y^2 < 1
bool point_in_area(const Point& p);

// check if polygon is inside the area D: x^2 + 4y^2 < 1
double polygon_in_area(vector<Point>& polygon);

// check if polygon is outside the area D: x^2 + 4y^2 < 1
double polygon_out_area(vector<Point>& polygon);

// function to find intersection of vectors and area boundary
Point find_intersect(const Point& p1, const Point& p2);

// get polygon P_ij for F_ij
vector<Point> get_Pij(Point& p, const double h1, const double h2);

// get S_ij = part of P_ij intersected with D
vector<Point> get_Sij(vector<Point>& P_ij);

// get L_ij = part of vec_ij intersected with D
vector<Point> get_Lij(vector<Point>& vec_ij);

//init F_ij array
template <size_t rows, size_t cols>
void init_F_arr(double (&F)[rows][cols], Point (&coords)[rows][cols], const double h1, const double h2)
{
    // f(x_i, y_j) = 1
    cout << "\n-------------Init F_ij-------------" << endl;
    for (size_t i = 1; i < rows - 1; i++)
        for (size_t j = 1; j < cols - 1; j++){
            vector<Point> P_ij = get_Pij(coords[i][j], h1, h2);
            // cout << "\n P_ij for (i, j): " << i << ", " << j << " (x_i, y_j): " << coords[i][j] << endl;
            // P_ij inside D area
            if (polygon_in_area(P_ij))
            {
                //cout << "\tP_ij is inside D:\n\t";
                F[i][j] = 1.0;
            }
            // P_ij outside D area
            else if (polygon_out_area(P_ij))
            {
                //cout << "\tP_ij is outside D:\n\t";
                F[i][j] = 0.0;
            }
            // P_ij intersect D area
            else
            {
                // cout << "\tP_ij intersects D:\n\t";
                // get S_ij
                vector<Point> S_ij = get_Sij(P_ij);
                // for (auto p: S_ij) { cout << p << ' '; }
                // cout << endl;
                F[i][j] = S_ij.size() ? polygon_area(S_ij) / (h1 * h2) : 0.0;
            }
            // cout << "\tF_ij: " << F[i][j] << endl;
        }       
}

//init a_ij array
template <size_t rows, size_t cols>
void init_coef_arr(double (&coef)[rows][cols], Point (&coords)[rows][cols], const double h1, const double h2, const char mode = 'a')
{
    vector<Point> vec_ij;
    const double epsilon = pow(max(h1, h2), 2);
    printf("\n-------------Init %c_ij-------------\n", mode);
    for (size_t i = 0; i < rows-1; i++)
        for (size_t j = 1; j < cols; j++){
            if (mode == 'a') { vec_ij = {Point(coords[i][j].x - h1/2, coords[i][j].y - h2/2), Point(coords[i][j].x - h1/2, coords[i][j].y + h2/2)}; }
            else { vec_ij = { Point(coords[i][j].x - h1/2, coords[i][j].y - h2/2), Point(coords[i][j].x + h1/2, coords[i][j].y - h2/2)}; }
            // cout << "\n vec_ij for (i, j): " << i << ", " << j << " (x_i, y_j): " << coords[i][j] << endl;
            // vec_ij inside D area
            if (polygon_in_area(vec_ij))
            {
                // cout << "\tvec_ij is inside D:\n\t";
                coef[i][j] = 1.0;
            }
            // vec_ij outside D area
            else if (polygon_out_area(vec_ij))
            {
                // cout << "\tvec_ij is outside D:\n\t";
                coef[i][j] = 1 / epsilon;
            }
            // vec_ij intersect D area
            else
            {
                // cout << "\tvec_ij intersects D:\n\t";
                // get l_ij
                vector<Point> l_ij = get_Lij(vec_ij);
                // cout << "\tGot l_ij:\n\t";
                // for (auto p: l_ij) { cout << p << ' '; }
                // cout << endl;
                coef[i][j] = 1 / epsilon;
                if (l_ij.size()) {
                    double len_l_ij = length(l_ij[0], l_ij[1]);
                    double h = mode == 'a' ? h2 : h1; // if A_ij then divide on h2, B_ij then h1
                    coef[i][j] = len_l_ij / h + (1 - len_l_ij / h) / epsilon;
                }
            }
            // printf("\t%c_ij: %.3f\n", mode, coef[i][j]);
        }         
}

//realization of operator A: Aw = B
template <size_t rows, size_t cols>
void operator_A(double (&res)[rows][cols], double (&w)[rows][cols], double (&a)[rows][cols], double (&b)[rows][cols], const double h1, const double h2)
{
    #pragma omp parallel for
    for (size_t i = rows - 2; i > 0; i--)
        for (size_t j = 1; j < cols - 1; j++)
        {
            double first_part = -1./h1 * (a[i][j+1] * (w[i][j+1] - w[i][j])/h1 - a[i][j] * (w[i][j] - w[i][j-1])/h1);
            double second_part = -1./h2 * (b[i-1][j] * (w[i-1][j] - w[i][j])/h2 - b[i][j] * (w[i][j] - w[i+1][j])/h2);
            res[i][j] = first_part + second_part;
        }          
}

// matrix subtrack: res = A - B
template <size_t rows, size_t cols>
void matrix_sub(double (&res)[rows][cols], double (&a)[rows][cols], double (&b)[rows][cols])
{
    #pragma omp parallel for
    for (size_t i = 1; i < rows - 1; i++)
        for (size_t j = 1; j < cols - 1; j++) {
            res[i][j] = a[i][j] - b[i][j];
        }
}

// matrix coef: res = coef * A
template <size_t rows, size_t cols>
void matrix_coef(double (&res)[rows][cols], double (&a)[rows][cols], const double coef)
{
    #pragma omp parallel for
    for (size_t i = 1; i < rows - 1; i++)
        for (size_t j = 1; j < cols - 1; j++) {
            res[i][j] = coef * a[i][j];
        }
}

template <size_t rows, size_t cols>
void matrix_copy(double (&res)[rows][cols], double (&a)[rows][cols])
{
    #pragma omp parallel for
    for (size_t i = 1; i < rows - 1; i++)
        for (size_t j = 1; j < cols - 1; j++) {
            res[i][j] = a[i][j];
        }
}