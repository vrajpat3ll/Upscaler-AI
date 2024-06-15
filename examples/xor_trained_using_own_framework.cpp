#include "../include/neural_network.hpp"
using namespace std;

float train_xor[]=
{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
float train_or[]=
{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};
float train_and[]=
{
    0, 0, 0,
    0, 1, 0,
    1, 0, 0,
    1, 1, 1,
};
float *trainingData = train_or;

int NSamples = 12 / 3;

int main() {
    srand(time(0));
    NeuralNetwork nn, g;
    vector<int> arch = {2, 2, 1};
    vector<float(*)(float)> funcs = {functions::ReLU, functions::sigmoidf};
    nn.init(arch, funcs);
    g.init(arch, funcs);
    nn.randomise(0, 1);

    matrix<> ti = matrix<>(NSamples, 2, 3, trainingData);
    matrix<> to = matrix<>(NSamples, 1, 3, trainingData + 2);
    matrix<> row = mat_row(ti, 0);
    MATRIX_PRINT(ti);
    MATRIX_PRINT(to);
    nn.input().copy(row);

    MATRIX_PRINT(row);
    MATRIX_PRINT(nn.input());
    // cout << "cost = " << nn.cost(ti, to) << endl;
    const float eps = 1e-1;
    const float rate = 1;
    const int epochs = 20 * 1000;

    for (int i = 0; i < epochs; i++) {
        nn.finite_diff(g, eps, ti, to);
        nn.learn(g, rate);
        // cout << "cost = " << nn.cost(ti, to) << endl;
    }
    cout << "\e[32mAfter-cost = \e[0m" << nn.cost(ti, to) << endl;
    nn.print("nn");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            nn.input().value(0, 0) = i;
            nn.input().value(0, 1) = j;
            nn.forward();
            cout << i << " ^ " << j << " = " << nn.output().value(0, 0) << endl;
        }
    }
}