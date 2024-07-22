#include <bits/stdc++.h>

#include "../include/neural_network.hpp"
#include "raylib.h"
using namespace std;

const int screenWidth = 16 * 100;
const int screenHeight = 10 * 100;

void NN_render_raylib(NeuralNetwork &nn, const vector<int> &arch) {
    Color backgorundColor = {0x16, 0x16, 0x16, 0xFF};  // greyish
    Color lowColor = {0xFF, 0x00, 0xFF, 0x00};
    Color highColor = {0x00, 0xFF, 0x00, 0x00};
    Color neuronColor = RED;
    Color connectionColor = GREEN;

   

    float neuronRadius = 020;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = screenWidth - 2 * layer_border_hpad;
    int layer_hpad = nn_width / arch.size();
    int nn_height = screenHeight - 2 * layer_border_vpad;
    int nn_x = screenWidth / 2 - nn_width / 2;
    int nn_y = screenHeight / 2 - nn_height / 2;
    for (int l = 0; l < arch.size(); l++) {
        int layer_vpad1 = nn_height / (arch[l]);
        for (int j = 0; j < arch[l]; j++) {
            int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + j * layer_vpad1 + layer_vpad1 / 2;
            if (l + 1 < arch.size()) {
                for (int k = 0; k < arch[l + 1]; k++) {
                    int layer_vpad2 = nn_height / (arch[l + 1]);
                    int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = nn_y + k * layer_vpad2 + layer_vpad2 / 2;
                    DrawLine(cx1, cy1, cx2, cy2, connectionColor);
                }
            }
            DrawCircle(cx1, cy1, neuronRadius, neuronColor);
        }
    }
    ClearBackground(backgorundColor);
}

float train_xor[] =
    {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
};
float train_or[] =
    {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 1,
};
float train_and[] =
    {
        0, 0, 0,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
};
float *trainingData = train_xor;

int NSamples = 12 / 3;

int main() {
    InitWindow(screenWidth, screenHeight, "Training sesh!");
    SetTargetFPS((int)1e6);  // don't cap fps

    srand(time(0));
    NeuralNetwork nn, g;
    vector<int> arch = {2, 2, 1};
    vector<float (*)(float)> funcs = {functions::ReLU, functions::sigmoidf};
    nn.init(arch, funcs);
    g.init(arch, funcs);
    nn.randomise(-1, 1);

    matrix<> ti = matrix<>(NSamples, 2, 3, trainingData);
    matrix<> to = matrix<>(NSamples, 1, 3, trainingData + 2);
    matrix<> row = mat_row(ti, 0);
    MATRIX_PRINT(ti);
    MATRIX_PRINT(to);
    nn.input().copy(row);

    const float eps = 1e-1;
    const float rate = 1e-1;
    const int epochs = 20 * 1000;
    int i = 1;
    while (!WindowShouldClose()) {
        if (i <= epochs) {
            i++;
            nn.finite_diff(g, eps, ti, to);
            nn.learn(g, rate);
            if (i % 100 == 0) std::cout << "\e[18;1H" << i << ": \e[31m" << nn.cost(ti, to) << "\n\e[0m";
        }
        BeginDrawing();
        NN_render_raylib(nn, arch);
        {
            string epoc = "epoch: " + to_string(i) + " / " + to_string(epochs);
            DrawText(epoc.c_str(), 0, 0, 18, WHITE);
            string fps = "FPS: " + to_string(GetFPS());
            DrawText(fps.c_str(), 0.94 * screenWidth, 0, 18, WHITE);
        }
        EndDrawing();
    }
    CloseWindow();

    std::cout << "\e[32mAfter-cost = \e[0m" << nn.cost(ti, to) << endl;
    nn.print("nn");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            nn.input().value(0, 0) = i;
            nn.input().value(0, 1) = j;
            nn.forward();
            std::cout << i << " ^ " << j << " = " << nn.output().value(0, 0) << endl;
        }
    }
}