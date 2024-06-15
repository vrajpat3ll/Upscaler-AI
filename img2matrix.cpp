#include <bits/stdc++.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
using namespace std;

char *args_shift(int &argc, char ***argv) {
    if (argc <= 0) {
        exit(1);
    }
    char *result = **argv;
    argc--;
    (*argv)++;
    return result;
}

int main(int argc, char **argv) {
    char *program = args_shift(argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "\e[37mUsage: %s <input.png>\e[0m\n", program);
        fprintf(stderr, "\e[31mError: No input file provided!\n\e[0m");
        return 1;
    }

    const char *img_file_path = args_shift(argc, &argv);

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
    cout << img_file_path << " size " << img_width << "x" << img_height << " " << img_components * 8 << " bits\n";

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
        }
    }
    MATRIX_PRINT(training_data);
    // const char *out_file_path = "img.mat";
    // FILE *out = fopen(out_file_path, "wb");
    // if (out == NULL) {
    //     fprintf(stderr, "<Matrix Saving> ERROR: Could not open file %s!\n", out_file_path);
    // }
    // // mat_save( out, training_data);
    // cout << "Generated " << out_file_path << " from " << img_file_path << "!\n";
}