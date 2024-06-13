#ifndef NN_H
#define NN_H
#include <math.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.hpp"

namespace functions {

float sigmoidf(float x) {
    return 1.f / (1 + exp(-x));
}
float ReLU(float x) {
    return ((x > 0) ? x : 0);
}
}  // namespace functions

class NeuralNetwork {
   private:
    //? <> means default parameters
    int count;
    float (*activation_function)(float x) = functions::sigmoidf;  // sigmoid by default
    std::vector<matrix<>> weights;
    std::vector<matrix<>> biases;
    std::vector<matrix<>> activations;  // count + 1 activations

   public:
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
    void init(const std::vector<int> &architecture);

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

    // TODO: implement void backprop();
};

#endif  // NN_H

#ifndef NN_IMPLMENTATION
#define NN_IMPLMENTATION

void NeuralNetwork::init(const std::vector<int> &architecture) {
    //? subtract 1 as 1st layer is input
    this->count = architecture.size() - 1;

    this->weights = std::vector<matrix<>>(this->count);
    this->biases = std::vector<matrix<>>(this->count);
    this->activations = std::vector<matrix<>>(this->count + 1);

    this->activations[0] = matrix<>(1, architecture[0]);
    this->activation_function = functions::sigmoidf;
    for (int i = 1; i <= this->count; i++) {
        this->weights[i - 1] = matrix(this->activations[i - 1].getCols(), architecture[i]);
        this->biases[i - 1] = matrix(1, architecture[i]);
        this->activations[i] = matrix(1, architecture[i]);
    }
}

void NeuralNetwork::print(const std::string &name) {
    std::cout << name << " = [" << std::endl;
    for (int i = 0; i < this->count; i++) {
        this->activations[i].print("a[" + std::to_string(i) + "]", 4);
        this->weights[i].print("w[" + std::to_string(i) + "]", 4);
        this->biases[i].print("b[" + std::to_string(i) + "]", 4);
    }
    this->activations[this->count].print("a[" + std::to_string(this->count) + "]", 4);
    std::cout << "]\n";
}

void NeuralNetwork::randomise(const float &low, const float &high) {
    for (int i = 0; i < this->count; i++) {
        this->weights[i].randomise(low, high);
        this->biases[i].randomise(low, high);
    }
}

void NeuralNetwork::forward() {
    for (int i = 0; i < this->count; i++) {
        mat_dot(this->activations[i + 1], this->activations[i], this->weights[i]);
        mat_sum(this->activations[i + 1], this->activations[i + 1], this->biases[i]);
        this->activations[i + 1].apply(this->activation_function);
    }
}

float NeuralNetwork::cost(matrix<> &trainingInput, matrix<> &trainingOutput) {
    if (trainingInput.getRows() != trainingOutput.getRows()) {
        fprintf(stderr, "\e[31m<Cost function> Input and Output's rows do not match!\e[0m");
        exit(1);
    }
    if (trainingOutput.getCols() != this->output().getCols()) {
        fprintf(stderr, "\e[31m<Cost function> Output and Output matrix's do not match!\e[0m");
        exit(1);
    }
    auto save = this->input();
    int n = trainingInput.getRows();
    float cost = 0;
    for (int i = 0; i < n; i++) {
        matrix<> x = mat_row(trainingInput, i);
        matrix<> y = mat_row(trainingOutput, i);

        this->input().copy(x);
        this->forward();
        int q = trainingOutput.getCols();
        for (int j = 0; j < q; j++) {
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

    for (int i = 0; i < this->count; i++) {
        for (int j = 0; j < this->weights[i].getRows(); j++) {
            for (int k = 0; k < this->weights[i].getCols(); k++) {
                saved = this->weights[i].value(j, k);

                this->weights[i].value(j, k) += eps;
                //? finite difference here, derivative could be faster in my opinion!
                g.weights[i].value(j, k) = (cost(ti, to) - c) / eps;

                this->weights[i].value(j, k) = saved;
            }
        }

        for (int j = 0; j < this->biases[i].getRows(); j++) {
            for (int k = 0; k < this->biases[i].getCols(); k++) {
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
    for (int i = 0; i < this->count; i++) {
        for (int j = 0; j < this->weights[i].getRows(); j++) {
            for (int k = 0; k < this->weights[i].getCols(); k++) {
                this->weights[i].value(j, k) -= rate * g.weights[i].value(j, k);
            }
        }
    }
    for (int i = 0; i < this->count; i++) {
        for (int j = 0; j < this->biases[i].getRows(); j++) {
            for (int k = 0; k < this->biases[i].getCols(); k++) {
                this->biases[i].value(j, k) -= rate * g.biases[i].value(j, k);
            }
        }
    }
}

#endif  // NN_IMPLMENTATION