#include "../include/neural_network.hpp"
using namespace std;

float trainingData[] =
{
    0,  0,
    1,  2,
    2,  4,
    3,  6,
    4,  8,
    5, 10,
    6, 12,
    7, 14,
    8, 16,
    9, 18,
    10,20,
};

int NSamples = sizeof(trainingData) / sizeof(trainingData[0]) / 2;

int main() {
    srand(time(0));
    NeuralNetwork nn, g;
    vector<int> arch = {1, 1};
    vector<float (*)(float)> funcs = {};
    nn.init(arch, funcs);
    g.init(arch, funcs);
    nn.randomise(-20, 20);

    matrix<> ti = matrix<>(NSamples, 1, 2, trainingData);
    matrix<> to = matrix<>(NSamples, 1, 2, trainingData + 1);
    matrix<> row = mat_row(ti, 0);
    MATRIX_PRINT(ti);
    MATRIX_PRINT(to);

    nn.print("nn");
    nn.input().copy(row);

    MATRIX_PRINT(row);
    MATRIX_PRINT(nn.input());
    // cout << "cost = " << nn.cost(ti, to) << endl;
    float eps = 1e-1;
    float rate = 1e-1;
    int epochs = 50 * 1000;
    for (int i = 1; i <= epochs; i++) {
        nn.finite_diff(g, eps, ti, to);
        nn.learn(g, rate);
        // cout << "cost = " << nn.cost(ti, to) << endl;
    }
    cout << "\e[32mAfter-cost = \e[0m" << nn.cost(ti, to) << endl;
    nn.print("nn");
    for (int i = 0; i < 21; i++) {
        nn.input().value(0, 0) = i;
        nn.forward();
        cout << i << " * 2 = " << nn.output().value(0, 0) << endl;
    }
}
