#include <bits/stdc++.h>

#include "raylib.h"
#include "include/neural_network.hpp"
using namespace functions;
using namespace std;

const int screenWidth = 16 * 100;
const int screenHeight = 10 * 100;

void NN_render_raylib(NeuralNetwork &nn, const vector<int> &arch) {
    Color backgorundColor = {0x18, 0x18, 0x18, 0xFF};  // greyish
    Color lowColor = {0xFF, 0x00, 0xFF, 0x00};
    Color highColor = {0x00, 0xFF, 0x00, 0x00};
    Color neuronColor = RED;
    Color connectionColor = GREEN;

    float neuronRadius = 20;
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

int main() {
    InitWindow(screenWidth, screenHeight, "Training sesh");
    SetTargetFPS(75);  // cap fps

    vector<int> arch = {4, 4, 2, 1};
    vector<float (*)(float)> acFs = {};
    NeuralNetwork nn, g;
    nn.init(arch, acFs);
    g.init(arch, acFs);
    nn.randomise(0, 1);
    nn.print("nn");
    float rate = 1;
    float eps = 1e-1;
    matrix<> ti;
    matrix<> to;
    int epoch = 1;
    int epochs = 5 * 1000;
    while (!WindowShouldClose()) {
        // if (epoch <= epochs) {
        //     nn.finite_diff(g, eps, ti, to);
        //     nn.learn(g, rate);
        //     epoch++;
        // }

        BeginDrawing();

        NN_render_raylib(nn, arch);

        EndDrawing();
    }
    CloseWindow();
    return 0;
}