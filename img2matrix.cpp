#include <bits/stdc++.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "include/external/stb_image.h"
using namespace std;

int main(int argc, char **argv) {

    if (argc <= 1) {
        fprintf(stderr, "\e[31mUsage: %s <input.png>\n"
                              "Error: No input file provided!\n\e[0m", argv[0]);
        return 1;
    }
    const char *img_file_path = argv[1];

    int img_width;
    int img_height;
    int img_components;

    unsigned char *img_pixels = (unsigned char *)stbi_load(img_file_path, &img_width, &img_height, &img_components, 0);
    if (img_pixels == NULL) {
        fprintf(stderr, "\e[31mError: Could not read image!\n%s\e[0m", img_file_path);
        return 1;
    }
    if (img_components != 1) {
        fprintf(stderr, "\e[31mError: %s is %d bits images!\nOnly 8 bit grayscale images are supported!\e[0m", img_file_path, img_components * 8);
        return 1;
    }
    cout << "\e[32m";
    cout << "[INFO] File path: " << img_file_path << '\n';
    cout << "[INFO] Size: " << img_width << "x" << img_height << '\n';
    cout << "[INFO] Bits: " << img_components * 8 << " bits\n\e[0m";

    matrix<> training_data(img_width * img_height, 3);  // x, y, intensity

    for (int y = 0; y < img_height; y++) {
        for (int x = 0; x < img_width; x++) {
            int i = y * img_width + x;
            float normalized_x = float(x) / (img_width - 1);
            float normalized_y = float(y) / (img_height - 1);
            float normalized_brightness = (float)img_pixels[i] / 255;

            training_data.value(i, 0) = normalized_x;
            training_data.value(i, 1) = normalized_y;
            training_data.value(i, 2) = normalized_brightness;
            // if ((int)img_pixels[i] == 0)
            //     cout << setw(3) << " " << ' ';
            // else
            //     cout << setw(3) << (int)img_pixels[i] << ' ';
        }
        // cout << '\n';
    }
    MATRIX_PRINT(training_data);
    string  storeLocation = "number.mat";
    training_data.save(storeLocation);
}