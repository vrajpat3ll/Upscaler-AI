#include <bits/stdc++.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "include/external/stb_image.h"
using namespace std;

typedef enum ErrorNo {
    INPUT_NOT_GIVEN,
    COULD_NOT_READ_IMAGE,
    IMAGE_AINT_8_BITS,
} ErrorNo;

int main(int argc, char **argv) {
    if (argc <= 1) {
        cerr << "\e[31mUsage: " << argv[0] << " <input.png>\n"
             << "Error: No input file provided!\n\e[0m";
        return INPUT_NOT_GIVEN;
    }
    const char *imgFile = argv[1];

    int imgWidth;
    int imgHeight;
    int imgComponents;

    unsigned char *img_pixels = (unsigned char *)stbi_load(imgFile, &imgWidth, &imgHeight, &imgComponents, 0);
    if (img_pixels == NULL) {
        cerr << "\e[31mError: Could not read image!\n"
             << imgFile << "\e[0m";
        return COULD_NOT_READ_IMAGE;
    }
    if (imgComponents != 1) {
        cerr << "\e[31mError: " << imgFile << "is" << imgComponents * 8 << "bits images!\n"
             << "Only 8 bit grayscale images are supported!\e[0m";
        return IMAGE_AINT_8_BITS;
    }
    cout << "\e[35m[INFO] File path: \e[33m" << imgFile << '\n';
    cout << "\e[35m[INFO] Size: \e[33m" << imgWidth << "x" << imgHeight << '\n';
    cout << "\e[35m[INFO] Bits: \e[33m" << imgComponents * 8 << " bits\n\e[0m";

    matrix<> trainingData(imgWidth * imgHeight, 3);  // x, y, intensity

    for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
            int i = y * imgWidth + x;
            float normalized_x = float(x) / (imgWidth - 1);
            float normalized_y = float(y) / (imgHeight - 1);
            float normalized_brightness = (float)img_pixels[i] / 255;

            trainingData.value(i, 0) = normalized_x;
            trainingData.value(i, 1) = normalized_y;
            trainingData.value(i, 2) = normalized_brightness;
            // if ((int)img_pixels[i] == 0)
            //     cout << setw(3) << " " << ' ';
            // else
            //     cout << setw(3) << (int)img_pixels[i] << ' ';
        }
        // cout << '\n';
    }

    // matrix<> ti(trainingData.getRows(), 2, trainingData.getCols(), &trainingData.value(0, 0));
    // matrix<> to(trainingData.getRows(), 1, trainingData.getCols(), &trainingData.value(0, ti.getCols()));

    // MATRIX_PRINT(trainingData);
    // MATRIX_PRINT(ti);
    // MATRIX_PRINT(to);
    string storeLocation; // = "image.mat";
    cout << "Enter path where to store the matrix: \n\e[33m";
    getline(cin, storeLocation);
    trainingData.save(storeLocation);
    cout << "\e[32mGenerated " << storeLocation << " from " << imgFile << "!\n\e[0m";
}