#include "neural_network.hpp"
using namespace std;
using namespace functions;


const int NSamples = 60*1000;

int main() {

    // input training matrix

    srand(time(0));

    NeuralNetwork nn, g;
    vector<int> arch = {28 * 28, 16, 10, 10};
    vector<float (*)(float)> activation_functions = {ReLU, sigmoidf, ReLU};
    nn.init(arch, activation_functions);
    g.init(arch, activation_functions);

    nn.randomise(-10, 10);

    matrix<> ti = matrix<>(NSamples * 28 * 28, 2);
    matrix<> to = matrix<>(NSamples * 28 * 28, 10);
    matrix<> row = mat_row(ti, 0);
    // MATRIX_PRINT(ti);
    // MATRIX_PRINT(to);
    // nn.input().copy(row);

    // MATRIX_PRINT(row);
    MATRIX_PRINT(nn.input());
    // cout << "cost = " << nn.cost(ti, to) << endl;
    const float eps = 1e-1;
    const float rate = 1;
    const int epochs = 20 * 1000;

    for (int i = 0; i < epochs; i++) {
        nn.finite_diff(g, eps, ti, to);
        nn.learn(g, rate);
        cout << "cost = " << nn.cost(ti, to) << endl;
    }
    // cout << "\e[32mAfter-cost = \e[0m" << nn.cost(ti, to) << endl;
    nn.print("nn");
    nn.forward();
    cout<<"After softmax\n";
    SoftMax(nn.output());
    
    nn.print("nn");
}