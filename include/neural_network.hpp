#ifndef NN_H
#define NN_H
#include <math.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.hpp"

#define DEBUG true

namespace functions {

float sigmoidf(float x) {
    return 1.f / (1 + exp(-x));
}
float ReLU(float x) {
    return ((x > 0) ? x : 0);
}
void SoftMax(matrix<> &m) {
    if (m.getCols() != 1 && m.getRows() != 1) {
        fprintf(stderr, "\e[31m<SoftMax function> Not a row or column matrix!\e[0m\n");
        exit(1);
    }
    float sum = 0;
    for (unsigned i = 0; i < m.getRows(); i++) {
        for (unsigned j = 0; j < m.getCols(); j++) {
            m.value(i, j) = expf(m.value(i, j));
            sum += m.value(i, j);
        }
    }
    for (unsigned i = 0; i < m.getRows(); i++) {
        for (unsigned j = 0; j < m.getCols(); j++) {
            m.value(i, j) /= sum;
        }
    }
}

int ArgMax(matrix<> &m) {
    if (m.getCols() != 1 && m.getRows() != 1) {
        fprintf(stderr, "\e[31m<SoftMax function> Not a row or column matrix!\e[0m\n");
        exit(1);
    }
    int ans = 0;
    if (m.getCols() == 1) {
        for (unsigned i = 0; i < m.getRows(); i++) {
            if (m.value(ans, 0) < m.value(i, 0)) ans = i;
        }
        return ans;
    }
    if (m.getRows() == 1) {
        for (unsigned i = 0; i < m.getCols(); i++) {
            if (m.value(0, ans) < m.value(0, i)) ans = i;
        }
        return ans;
    }
    return 0;
}
float derivative(float (*function)(float), float a) {
    if (function == functions::sigmoidf) {
        return function(a) * (1 - function(a));
    } else if (function == functions::ReLU) {
        return a >= 0;
    }
    return 0;
}
}  // namespace functions

class NeuralNetwork {
   private:
    unsigned count;
    std::vector<float (*)(float)> activation_functions;
    //? <> means default parameters
    std::vector<matrix<>> weights;
    std::vector<matrix<>> biases;
    std::vector<matrix<>> activations;  // count + 1 activations

   public:
    float &value(const std::string &option, unsigned layer, unsigned row, unsigned col);

    /// @brief evaluates the cost / loss function based on the inout and the output
    /// @param trainingInput input data
    /// @param trainingOutput output data
    /// @return J ( Theta )
    float cost(matrix<> &trainingInput, matrix<> &trainingOutput);

    /// @return a reference to the input matrix
    matrix<> &input() { return this->activations[0]; }

    /// @return a reference to the output matrix
    matrix<> &output() { return this->activations[this->count]; }

    /// @brief initialize the neural network.
    /// First element must contain the number of inputs.
    /// @param architecture a vector containing the number of neurons in each layer.
    void init(const std::vector<unsigned> &architecture, const std::vector<float (*)(float)> &activation_functions);

    /// @brief Adjusts the NeuralNetwork to better-fit the data
    /// @param gradient a neural network containing the difference to be made for
    /// each neuron in the NeuralNetwork
    /// @param rate Rate of learning
    void learn(NeuralNetwork &gradient /*difference*/, float rate);

    void print(const std::string &name);
    void randomise(const float &low, const float &high);

    /// @brief forwards the input and stores the output in
    /// this -> output( )
    void forward();

    /// @param g an empty NeuralNetwork of the same architecture
    /// @param eps epsilon/ delta/
    /// @param trainingInput input data
    /// @param trainingOutput output data
    void finite_diff(NeuralNetwork &g, const float &eps, matrix<> &trainingInput, matrix<> &trainingOutput);

    // TODO: implement
    void backpropagate(NeuralNetwork &g, matrix<> &input, matrix<> &target);
    friend void nn_zero(NeuralNetwork &nn);
};

#endif  // NN_H

#ifndef NN_IMPLMENTATION
#define NN_IMPLMENTATION

void NeuralNetwork::init(const std::vector<unsigned> &architecture, const std::vector<float (*)(float)> &activation_functions) {
    //? subtract 1 as 1st layer is input
    this->count = architecture.size() - 1;
    bool no_activation_functions = (activation_functions.size() == 0);
    if (no_activation_functions) goto label;

    if (this->count != activation_functions.size()) {
        fprintf(stderr, "\e[31m<Neural Network Init> The number of layers and the number of activation functions are not the same!\e[0m\n");
        exit(1);
    }
label:
    this->weights = std::vector<matrix<>>(this->count);
    this->biases = std::vector<matrix<>>(this->count);
    this->activations = std::vector<matrix<>>(this->count + 1);
    this->activation_functions = std::vector<float (*)(float)>(this->count, functions::sigmoidf);

    this->activations[0] = matrix<>(1, architecture[0]);
    for (unsigned i = 1; i <= this->count; i++) {
        this->weights[i - 1] = matrix(this->activations[i - 1].getCols(), architecture[i]);
        this->biases[i - 1] = matrix(1, architecture[i]);
        this->activations[i] = matrix(1, architecture[i]);
    }
    if (no_activation_functions) goto end;

    std::copy(activation_functions.begin(), activation_functions.end(), this->activation_functions.begin());
end:;
}

void NeuralNetwork::print(const std::string &name) {
    std::cout << name << " = [" << std::endl;
    for (unsigned i = 0; i < this->count; i++) {
        this->activations[i].print("a[" + std::to_string(i) + "]", 4);
        this->weights[i].print("w[" + std::to_string(i) + "]", 4);
        this->biases[i].print("b[" + std::to_string(i) + "]", 4);
    }
    this->activations[this->count].print("a[" + std::to_string(this->count) + "]", 4);
    std::cout << "]\n";
}

void NeuralNetwork::randomise(const float &low, const float &high) {
    for (unsigned i = 0; i < this->count; i++) {
        this->weights[i].randomise(low, high);
        this->biases[i].randomise(low, high);
    }
}

void NeuralNetwork::forward() {
    for (unsigned i = 0; i < this->count; i++) {
        mat_dot(this->activations[i + 1], this->activations[i], this->weights[i]);
        mat_sum(this->activations[i + 1], this->activations[i + 1], this->biases[i]);
        this->activations[i + 1].apply(this->activation_functions[i]);
    }
}

float &NeuralNetwork::value(const std::string &option, unsigned layer, unsigned row, unsigned col) {
    std::string log;
    if (layer >= count) {
        log = "\e[31m<value> Layer index exceeded!\n\e[0m";
        fprintf(stderr, log.c_str());
        throw std::runtime_error(log);
    }
    if (option == "weights") {
        if (row >= this->weights[layer].getRows()) {
            log = "\e[31m<value> Weights at layer " + std::to_string(layer) + " exceeded row count!\n\e[0m";
            fprintf(stderr, log.c_str());
            throw std::runtime_error(log);
        }
        if (col >= this->weights[layer].getCols()) {
            log = "\e[31m<value> Weights at layer " + std::to_string(layer) + " exceeded col count!\n\e[0m";
            fprintf(stderr, log.c_str());
            throw std::runtime_error(log);
        }
        return this->weights[layer].value(row, col);
    } else if (option == "biases") {
        if (row >= this->biases[layer].getRows()) {
            log = "\e[31m<value> Biases at layer " + std::to_string(layer) + " exceeded row count!\n\e[0m";
            fprintf(stderr, log.c_str());
            throw std::runtime_error(log);
        }
        if (col >= this->biases[layer].getCols()) {
            log = "\e[31m<value> Biases at layer " + std::to_string(layer) + " exceeded col count!\n\e[0m";
            fprintf(stderr, log.c_str());
            throw std::runtime_error(log);
        }
        return this->biases[layer].value(row, col);
    } else if (option == "activations") {
        if (row >= this->activations[layer].getRows()) {
            log = "\e[31m<value> Activations at layer " + std::to_string(layer) + " exceeded row count!\n\e[0m";
            fprintf(stderr, log.c_str());
            throw std::runtime_error(log);
        }
        if (col >= this->activations[layer].getCols()) {
            log = "\e[31m<value> Activations at layer " + std::to_string(layer) + " exceeded col count!\n\e[0m";
            fprintf(stderr, log.c_str());
            throw std::runtime_error(log);
        }
        return this->activations[layer].value(row, col);
    } else {
        fprintf(stderr, "\e[31m<value> Wrong option given!\nChoose from \"weights\", \"biases\" and \"activations\"\n\e[0m");
        throw std::runtime_error("\e[31m<value> Wrong option given!\nChoose from \"weights\", \"biases\" and \"activations\"\n\e[0m");
    }
}

float NeuralNetwork::cost(matrix<> &trainingInput, matrix<> &trainingOutput) {
    if (trainingInput.getRows() != trainingOutput.getRows()) {
        fprintf(stderr, "\e[31m<Cost function> Input and Output's rows do not match!\e[0m");
        exit(1);
    }
    if (trainingOutput.getCols() != this->output().getCols()) {
        fprintf(stderr, "\e[31m<Cost function> Output and Output matrix's cols do not match!\e[0m");
        exit(1);
    }
    auto save = this->input();
    unsigned n = trainingInput.getRows();
    float cost = 0;
    for (unsigned i = 0; i < n; i++) {
        matrix<> x = mat_row(trainingInput, i);
        matrix<> y = mat_row(trainingOutput, i);

        this->input().copy(x);
        this->forward();
        unsigned q = trainingOutput.getCols();
        for (unsigned j = 0; j < q; j++) {
            float diff = this->output().value(0, j) - y.value(0, j);
            cost += diff * diff;
        }
    }
    this->input() = save;
    return cost / (2 * n);  //? dividing by 2 just bcoz of convention
}

void NeuralNetwork::finite_diff(NeuralNetwork &g, const float &eps, matrix<> &ti, matrix<> &to) {
    float saved;
    float c = cost(ti, to);

    for (unsigned i = 0; i < this->count; i++) {
        for (unsigned j = 0; j < this->weights[i].getRows(); j++) {
            for (unsigned k = 0; k < this->weights[i].getCols(); k++) {
                saved = this->weights[i].value(j, k);

                this->weights[i].value(j, k) += eps;
                //? finite difference here, derivative could be faster in my opinion!
                g.weights[i].value(j, k) = (cost(ti, to) - c) / eps;

                this->weights[i].value(j, k) = saved;
            }
        }

        for (unsigned j = 0; j < this->biases[i].getRows(); j++) {
            for (unsigned k = 0; k < this->biases[i].getCols(); k++) {
                saved = this->biases[i].value(j, k);

                this->biases[i].value(j, k) += eps;
                //? finite difference here, derivative could be faster in my opinion!
                g.biases[i].value(j, k) = (cost(ti, to) - c) / eps;

                this->biases[i].value(j, k) = saved;
            }
        }
    }
}

void NeuralNetwork::learn(NeuralNetwork &g, float rate) {
    for (unsigned i = 0; i < this->count; i++) {
        for (unsigned j = 0; j < this->weights[i].getRows(); j++) {
            for (unsigned k = 0; k < this->weights[i].getCols(); k++) {
                this->weights[i].value(j, k) -= rate * g.weights[i].value(j, k);
            }
        }
    }
    for (unsigned i = 0; i < this->count; i++) {
        for (unsigned j = 0; j < this->biases[i].getRows(); j++) {
            for (unsigned k = 0; k < this->biases[i].getCols(); k++) {
                this->biases[i].value(j, k) -= rate * g.biases[i].value(j, k);
            }
        }
    }
}

// void nn_zero(NeuralNetwork &nn) {
//     for (size_t i = 0; i < nn.count; ++i) {
//         nn.weights[i].fill(0);
//         nn.biases[i].fill(0);
//         nn.activations[i].fill(0);
//     }
//     nn.activations[nn.count].fill(0);
// }

// void NeuralNetwork::backpropagate(NeuralNetwork &g, matrix<> &input, matrix<> &target) {
//     if (input.getRows() != target.getRows()) {
//         std::cout << "" << std::endl;
//         exit(1);
//     };
//     size_t n = input.getRows();
//     if (this->output().getCols() != target.getCols()) {
//         std::cout << "" << std::endl;
//         exit(1);
//     };

//     nn_zero(g);

//     // i - current sample
//     // l - current layer
//     // j - current activation
//     // k - previous activation

//     for (size_t i = 0; i < n; ++i) {
//         this->input().copy(mat_row(input, i));
//         this->forward();

//         // for (size_t j = 0; j <= this->count; ++j) {
//         //     mat_fill(g.as[j], 0);
//         // }

//         for (size_t j = 0; j < target.getCols(); ++j) {
//             g.output().value(0, j) = this->output().value(0, j) - target.value(i, j);
//         }

//         for (size_t l = this->count; l > 0; --l) {
//             for (size_t j = 0; j < this->activations[l].getCols(); ++j) {
//                 float a = this->activations[l].value(0, j);
//                 float da = g.activations[l].value(0, j);
//                 g.biases[l - 1].value(0, j) += 2 * da * a * (1 - a);
//                 for (size_t k = 0; k < this->activations[l - 1].getCols(); ++k) {
//                     // j - weight matrix col
//                     // k - weight matrix row
//                     float pa = this->activations[l - 1].value(0, k);
//                     float w = this->weights[l - 1].value(k, j);
//                     g.weights[l - 1].value(k, j) += 2 * da * a * (1 - a) * pa;
//                     g.activations[l - 1].value(0, k) += 2 * da * a * (1 - a) * w;
//                 }
//             }
//         }
//     }

//     for (size_t i = 0; i < this->count; ++i) {
//         for (size_t j = 0; j < g.weights[i].getRows(); ++j) {
//             for (size_t k = 0; k < g.weights[i].getCols(); ++k) {
//                 g.weights[i].value(j, k) /= n;
//             }
//         }
//         for (size_t j = 0; j < g.biases[i].getRows(); ++j) {
//             for (size_t k = 0; k < g.biases[i].getCols(); ++k) {
//                 g.biases[i].value(j, k) /= n;
//             }
//         }
//     }
// }

#endif  // NN_IMPLMENTATION