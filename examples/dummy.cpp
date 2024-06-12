#define MATRIX_IMPLEMENTATION
#include <math.h>
#include <time.h>

#include "../include/neural_network.hpp"
float sigmoidf(float x) {
    return 1.f / (1 + exp(-x));
}
float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0};

int NSamples = sizeof(td) / sizeof(td[0]) / 3;

int main() {
    srand(time(0));  // for creating random values all the time
    matrix<float> A, B;
    matrix<float> C;
    A.fill(0);
    A.apply(sigmoidf);
    B.randomise(0, 10);
    MATRIX_PRINT(A);
    MATRIX_PRINT(B);
    mat_sum(C, A, B);
    // A+=B;
    // B = B * A;
    std::cout << "After :\n";
    MATRIX_PRINT(C);
}