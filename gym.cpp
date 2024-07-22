/**
 * GUI app for training the neural networks.
 * Spit out a binary file which can be parsed to generate the neural network
 **/
#include <bits/stdc++.h>

#include "include/neural_network.hpp"
#include "raylib.h"
using namespace functions;
using namespace std;

float train_xor[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
float *trainingData = train_xor;

int NSamples = 12 / 3;
// matrix<> ti = matrix<>(NSamples, 2, 3, trainingData);
// matrix<> to = matrix<>(NSamples, 1, 3, trainingData + 2);

#define PRINT_FILES
#define RATE 1e-1
#define EPSILLON 1e-1
#define EPOCHS 20 * 1000
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
    cout << "\e[32mActivation functions parsed successfully\e[0m" << endl;
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
    cout << "\e[32mArchitecture parsed successfully\e[0m" << endl;
    return architecture;
}
};  // namespace parse

void NN_render_raylib(NeuralNetwork &nn, const vector<int> &arch) {
    Color backgroundColor = {0x18, 0x18, 0x18, 0xFF};  // greyish
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
    ClearBackground(backgroundColor);
}

// need to write own validate for each problem
void validate(NeuralNetwork &nn, matrix<> &ti, matrix<> &to) {
    int cnt = 0;
    for (int i = 0; i < ti.getRows(); i++) {
        nn.input() = mat_row(ti, i);
        nn.forward();
        matrix<> output = nn.output();
        std::cout << ti.value(i, 0) << " ^ " << ti.value(i, 1) << " = " << nn.output().value(0, 0) << endl;
        output.apply(round);
        matrix<> expected = mat_row(to, i);
        if (output == expected) cnt++;
    }

    float accuracy = (float)cnt / ti.getRows() * 100;
    std::cout << "accuracy: " << accuracy << "%" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "\e[31mUsage: gym ";
        cout << "<filepath to architecture> ";
        cout << "<filepath to activation functions>\e[0m";
        return -1;
    }

    vector<int> arch = parse::parseArchitecture(argv[1]);

    if (arch.size() < 3) {
        fprintf(stderr, "\e[31m<GYM> Architecture does not contain hidden layers?\n\e[0m");
        return 1;
    }
    vector<float (*)(float)> acFs;
    if (argc < 3) {
        acFs = {};
    } else {
        acFs = parse::parseActivationFunctions(argv[2]);
    }

    std::string inputData ;// = "C:\\CODING\\code\\projects\\C-C++\\Upscaler-AI\\Upscaler-AI\\examples\\testing\\save-method\\ti.mat";
    std::string outputData;// = "C:\\CODING\\code\\projects\\C-C++\\Upscaler-AI\\Upscaler-AI\\examples\\testing\\save-method\\to.mat";
    std::cout << "Enter path for input data (.mat format):\n";
    getline(cin, inputData);
    ifstream f(inputData);
    if (!f.is_open()) {
        fprintf(stderr, "\e[31m<GYM> Could not find input data file!\n\e[0m");
        return 1;
    }
    f.close();
    std::cout << "Enter path for output data (.mat format):\n";
    getline(cin, outputData);
    ifstream file(outputData);
    if (!file.is_open()) {
        fprintf(stderr, "\e[31m<GYM> Could not find output data file!\n\e[0m");
        return 1;
    }
    file.close();

    InitWindow(screenWidth, screenHeight, "Training sesh");
    SetTargetFPS(int(1e6));  // cap fps

    matrix<> ti;// = matrix<>(NSamples, 2, 3, trainingData);
    matrix<> to;// = matrix<>(NSamples, 1, 3, trainingData + 2);
    ti.load(inputData);
    to.load(outputData);
    if (ti.getRows() != to.getRows()) {
        fprintf(stderr, "\e[31m<GYM> Number of samples in input and output do not match!\n\e[0m");
        return 1;
    }
    if (ti.getCols() != arch[0]) {
        fprintf(stderr, "\e[31m<GYM> Number of features in input and architecture do not match!\n\e[0m");
        return 1;
    }
    if (to.getCols() != arch.back()) {
        fprintf(stderr, "\e[31m<GYM> Number of features in output and architecture do not match!\n\e[0m");
        return 1;
    }
#ifdef PRINT_FILES
    {
        cout << "\nArchitecture: ";
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
    MATRIX_PRINT(ti);
    MATRIX_PRINT(to);
    srand(time(0));
    NeuralNetwork nn, g;
    nn.init(arch, acFs);
    g.init(arch, acFs);
    nn.randomise(0, 1);
    nn.print("Neural Network");
    int epoch = 1;

    while (!WindowShouldClose()) {
        if (epoch <= EPOCHS) {
            nn.finite_diff(g, EPSILLON, ti, to);
            nn.learn(g, RATE);
            if (epoch % 100 == 0) std::cout << "\e[18;1H" << epoch << ": \e[31m" << nn.cost(ti, to) << "\n\e[0m";
            epoch++;
        }
        BeginDrawing();
        NN_render_raylib(nn, arch);
        {
            string epoc = "epoch: " + to_string(epoch) + " / " + to_string(EPOCHS);
            DrawText(epoc.c_str(), 0, 0, 18, WHITE);
            string fps = "FPS: " + to_string(GetFPS());
            DrawText(fps.c_str(), 0.94 * screenWidth, 0, 18, WHITE);
        }
        EndDrawing();
    }
    cout << "\e[19;1H";
    CloseWindow();
    cout << "\n";
    nn.print("Neural Network");

    validate(nn, ti, to);
    // nn.save("filename");
    return 0;
}