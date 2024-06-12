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
    // std::vector<int> arch = {2, 2, 1};
    // NeuralNetwork nn, g;
    // nn.init(arch);
    // g.init(arch);
    // nn.randomise(0, 1);
    // std::cout << "Init:\n";
    // nn.print("nn");
    // std::cout << "\n--------------------------\n\n";
    // matrix<> ti(NSamples, 2, 3, td);
    // matrix<> to(NSamples, 1, 3, td + 2);
    // matrix<> row = mat_row(ti, 2);
    // // MATRIX_PRINT(ti);
    // // MATRIX_PRINT(to);
    // MATRIX_PRINT(row);

    // nn.input().copy(row);
    // std::cout << "After: \n";

    // nn.forward();
    // nn.print("nn");
    // MATRIX_PRINT(nn.output());

    // return 0;

    srand(time(0));  // for creating random values all the time
    matrix<float> A(3, 3), B(3, 3);
    matrix<float> C(3, 3);
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