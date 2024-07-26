#define MATRIX_IMPLEMENTATION
#include "../../../include/matrix.hpp"

float arr[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

int main() {
    matrix<> t(4, 3, 3, arr);
    MATRIX_PRINT(t);
    t.save("t.mat");
    return 0;
}