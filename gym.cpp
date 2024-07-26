/**
 * GUI app for training the neural networks.
 * Spit out a binary file which can be parsed to generate the neural network
 * 
 **/
#include <bits/stdc++.h>

#include "include/neural_network.hpp"
#include "raylib.h"
using namespace functions;
using namespace std;

#define PRINT_FILES
#define FPS 1500
#define RATE 1e-1
#define EPSILLON 1e-1
#define EPOCHS 20 * 1000
const int screenWidth = 16 * 100;
const int screenHeight = 9 * 100;

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

vector<unsigned> parseArchitecture(string filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cout << "Could not open file" << endl;
        return {};
    }

    vector<unsigned> architecture;
    unsigned value;
    while (file >> value) {
        architecture.push_back(value);
    }

    file.close();
    cout << "\e[32mArchitecture parsed successfully\e[0m" << endl;
    return architecture;
}
};  // namespace parse

void NN_render_raylib(NeuralNetwork &nn, const vector<unsigned> &arch, int rx, int ry, int rw, int rh) {
    Color backgroundColor = {0x18, 0x18, 0x18, 0xFF};  // greyish
    Color lowColor = RED;
    Color highColor = DARKBLUE;

    float neuronRadius = rh * 0.03;
    int layer_border_vpad = rh * 0.08;
    int layer_border_hpad = rw * 0.06;
    int nn_width = rw - 2 * layer_border_hpad;
    int layer_hpad = nn_width / arch.size();
    int nn_height = rh - 2 * layer_border_vpad;
    int nn_x = rx + rw / 2 - nn_width / 2;
    int nn_y = ry + rh / 2 - nn_height / 2;
    for (unsigned l = 0; l < arch.size(); l++) {
        int layer_vpad1 = nn_height / (arch[l]);
        for (unsigned j = 0; j < arch[l]; j++) {
            int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            int cy1 = nn_y + j * layer_vpad1 + layer_vpad1 / 2;
            if (l + 1 < arch.size()) {
                for (unsigned k = 0; k < arch[l + 1]; k++) {
                    int layer_vpad2 = nn_height / (arch[l + 1]);
                    int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = nn_y + k * layer_vpad2 + layer_vpad2 / 2;
                    Vector2 start = {(float)cx1, (float)cy1};
                    Vector2 end = {(float)cx2, (float)cy2};
                    float value = sigmoidf(nn.value("weights", l, j, k));
                    highColor.a = floorf(255.0F * value);
                    float thickness = rh * 0.015F * (sigmoidf(abs(nn.value("weights", l, j, k)))-0.5);
                    DrawLineEx(start, end, thickness, ColorAlphaBlend(lowColor, highColor, WHITE));
                }
            }
            if (l > 0) {
                highColor.a = floor(255.0F * sigmoidf(nn.value("biases", l - 1, 0, j)));
                DrawCircle(cx1, cy1, neuronRadius, ColorAlphaBlend(lowColor, highColor, WHITE));
            } else 
                DrawCircle(cx1, cy1, neuronRadius, GRAY);
        }
    }
    ClearBackground(backgroundColor);
}

// need to write own validate for each problem
void validate(NeuralNetwork &nn, matrix<> &ti, matrix<> &to) {
    int cnt = 0;
    for (unsigned i = 0; i < ti.getRows(); i++) {
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
        cout << "<filepath to activation functions> "; 
        cout << "<filepath to training data> \e[0m";
        return -1;
    }

    vector<unsigned> arch = parse::parseArchitecture(argv[1]);

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
    
    std::string data;
    if (argc == 4) {
        data = argv[3];
    } else {
        std::cout << "Enter path for data (.mat format):\n";
        getline(cin, data);
    }
    ifstream f(data);
    if (!f.is_open()) {
        fprintf(stderr, "\e[31m<GYM> Could not find data file!\n\e[0m");
        return 1;
    }
    f.close();
    
    matrix t;
    t.load(data);
    if (arch[0] + arch.back() != t.getCols()) {
        fprintf(stderr, "\e[31m<Check> architecture and data's dimensions do not match!\n"
                               "architecture[0]    = %d\n"
                               "architecture[last] = %d\n"
                               "trainingData.cols  = %d\e[0m", arch[0], arch.back(), t.getCols());
        return 1;
    }
    matrix<> ti (t.getRows(), arch[0], t.getCols(), &t.value(0, 0));
    matrix<> to (t.getRows(), arch.back(), t.getCols(), &t.value(0, arch[0]));

    SetConfigFlags(FLAG_WINDOW_RESIZABLE); // to make the window resize-able
    InitWindow(screenWidth, screenHeight, "Training sesh");
    SetTargetFPS(FPS);
    // I guess these conditions are useless now!
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
    int epoch = 1;

    bool pause = true;
    while (!WindowShouldClose()) {
        
        if (epoch <= EPOCHS && !pause) {
            nn.finite_diff(g, EPSILLON, ti, to);
            nn.learn(g, RATE);
            if (epoch % 100 == 0)
                std::cout << "\e[18;1H" << epoch << ": \e[31m" << nn.cost(ti, to) << "\n\e[0m";
            epoch++;
        }
        if (IsKeyPressed(KEY_R)) {
            srand(time(0));
            nn.randomise(0, 1);
            epoch = 0;
        }
        if (IsKeyPressed(KEY_SPACE)) {
            pause = !pause;
        }
        BeginDrawing();
        {
            int rx, ry, rw, rh;
            
            rw = GetScreenWidth() / 2;
            rh = GetScreenHeight() * 3 / 4;
            rx = 0;
            ry = 0;

            NN_render_raylib(nn, arch, rx, ry, rw, rh);

            string epoc = "epoch: " + to_string(epoch) + " / " + to_string(EPOCHS);
            string cost = "cost: " + to_string(nn.cost(ti, to));
            DrawText(epoc.c_str(), 0, 0, 24, WHITE);
            DrawText(cost.c_str(), GetScreenWidth() / 2, 0, 24, WHITE);
            string fps = "FPS: " + to_string(GetFPS());
            DrawText(fps.c_str(), GetScreenWidth() - fps.size() * 13, 0, 24, WHITE);
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