#define MATRIX_IMPLEMENTATION
#include "../../../include/matrix.hpp"

float arr[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

int main() {
    matrix<> ti(4, 2, 3, arr);
    matrix<> to(4, 1, 3, arr+3);
    MATRIX_PRINT(ti);
    MATRIX_PRINT(to);
    ti.save("ti.mat");
    to.save("to.mat");
    return 0;
}