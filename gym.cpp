/**
 * GUI app for training the neural networks.
 * Spit out a binary file which can be parsed to generate the neural network
 **/
#include <bits/stdc++.h>

#include "include/neural_network.hpp"
#include "raylib.h"
using namespace functions;
using namespace std;

#define PRINT_FILES

const int screenWidth = 16 * 100;
const int screenHeight = 10 * 100;

namespace parse {
vector<float (*)(float)> parseActivationFunctions(string filepath) {
    vector<float (*)(float)> activationFunctions;
    ifstream file(filepath);

    if (!file.is_open()) {
        cout << "Could not open activation functions file" << endl;
        return activationFunctions;
    }

    string functionName;
    while (file >> functionName) {
        if (functionName == "ReLU") {
            activationFunctions.push_back(functions::ReLU);
        } else if (functionName == "sigmoidf") {
            activationFunctions.push_back(functions::sigmoidf);
        }
        // Add more activation functions as needed
    }

    file.close();
    cout << "Activation functions parsed successfully" << endl;
    return activationFunctions;
}

vector<int> parseArchitecture(string filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cout << "Could not open file" << endl;
        return {};
    }

    vector<int> architecture;
    int value;
    while (file >> value) {
        architecture.push_back(value);
    }

    file.close();
    cout << "Architecture parsed successfully" << endl;
    return architecture;
}
};  // namespace parse

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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "\e[31m<exe input> Usage: gym ";
        cout << "<filepath to architecture> ";
        cout << "<filepath to activation functions>\e[0m";
        return -1;
    }

    vector<int> arch = parse::parseArchitecture(argv[1]);
    // vector<int> arch = parse::parseArchitecture("c:/coding/code/projects/c-c++/upscaler-ai/upscaler-ai/network.arch");
    vector<float (*)(float)> acFs;
    if (argc < 3) {
        acFs = {};
    } else
        acFs = parse::parseActivationFunctions(argv[2]);

#ifdef PRINT_FILES
    {
        cout << "Architecture: ";
        for (auto x : arch) cout << x << " ";
        cout << endl;
        cout << "Activations: ";
        for (auto x : acFs) {
            string name;
            if (x == functions::ReLU)
                name = "ReLU";
            else if (x == functions::sigmoidf)
                name = "sigmoidf";

            cout << name << " ";
        }
        cout << endl;
    }
#endif
    InitWindow(screenWidth, screenHeight, "Training sesh");
    SetTargetFPS(75);  // cap fps
    NeuralNetwork nn, g;
    nn.init(arch, acFs);
    g.init(arch, acFs);
    nn.randomise(0, 1);
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
        {
            string epoc = "epoch: " + to_string(epoch) + " / " + to_string(epochs);
            DrawText(epoc.c_str(), 0, 0, 18, WHITE);
            string fps = "FPS: " + to_string(GetFPS());
            DrawText(fps.c_str(), 0.94 * screenWidth, 0, 18, WHITE);
        }

        EndDrawing();
    }
    CloseWindow();

    nn.print("Neural Network");
    // nn.save("filename");
    return 0;
}