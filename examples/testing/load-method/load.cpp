#define MATRIX_IMPLEMENTATION
#include "../../../include/matrix.hpp"

int main() {
    matrix ti, to;
    ti.load("../save-method/ti.mat");
    to.load("../save-method/to.mat");
    MATRIX_PRINT(ti);
    MATRIX_PRINT(to);
    return 0;
}