#define NN_IMPLEMENTATION
#include "../include/neural_network.hpp"

struct Xor {
    matrix<float>* a0 = new matrix(1, 2);
    matrix<float>* w1 = new matrix(2, 2);
    matrix<float>* b1 = new matrix(1, 2);
    matrix<float>* a1 = new matrix(1, 2);
    matrix<float>* w2 = new matrix(2, 1);
    matrix<float>* b2 = new matrix(1, 1);
    matrix<float>* a2 = new matrix(1, 1);
};

// Xor xor_alloc() {
//     Xor m;
//     // m.a0
//     // m.w1
//     // m.b1
//     // m.a1
//     // m.w2
//     // m.b2
//     // m.a2
//     return m;
// }

void forward_xor(Xor& m) {
    mat_dot(*m.a1, *m.a0, *m.w1);
    mat_sum(*m.a1, *m.a1, *m.b1);
    m.a1->apply(functions::sigmoidf);

    mat_dot(*m.a2, *m.a1, *m.w2);
    mat_sum(*m.a2, *m.a2, *m.b2);
    m.a2->apply(functions::sigmoidf);
}

template <typename T>
float cost(Xor model, matrix<T> trainingInput, matrix<T> trainingOutput) {
    if (trainingInput.getRows() != trainingOutput.getRows()) {
        fprintf(stderr, "\e[31m<Cost function> Input and Output's rows do not match!\e[0m");
        exit(1);
    }
    if (trainingOutput.getCols() != model.a2->getCols()) {
        fprintf(stderr, "\e[31m<Cost function> Output and Output matrix's do not match!\e[0m");
        exit(1);
    }

    int n = trainingInput.getRows();
    T cost = 0;
    for (int i = 0; i < n; i++) {
        matrix<> x = mat_row(trainingInput, i);
        matrix<> y = mat_row(trainingOutput, i);

        model.a0->copy(x);
        forward_xor(model);

        int q = trainingOutput.getCols();
        for (int j = 0; j < q; j++) {
            T diff = model.a2->value(0, j) - y.value(0, j);
            cost += diff * diff;
        }
    }
    return cost / (2 * n);  // dividing by 2 just bcoz of convention
}

void xor_learn(Xor m, Xor g, float rate) {
    for (int i = 0; i < m.w1->getRows(); i++) {
        for (int j = 0; j < m.w1->getCols(); j++) {
            m.w1->value(i, j) -= rate * g.w1->value(i, j);
        }
    }
    for (int i = 0; i < m.b1->getRows(); i++) {
        for (int j = 0; j < m.b1->getCols(); j++) {
            m.b1->value(i, j) -= rate * g.b1->value(i, j);
        }
    }
    for (int i = 0; i < m.w2->getRows(); i++) {
        for (int j = 0; j < m.w2->getCols(); j++) {
            m.w2->value(i, j) -= rate * g.w2->value(i, j);
        }
    }
    for (int i = 0; i < m.b2->getRows(); i++) {
        for (int j = 0; j < m.b2->getCols(); j++) {
            m.b2->value(i, j) -= rate * g.b2->value(i, j);
        }
    }
}

void finite_diff(Xor m, Xor g, float eps, matrix<> ti, matrix<> to) {
    float saved;
    float c = cost(m, ti, to);
    for (int i = 0; i < m.w1->getRows(); i++) {
        for (int j = 0; j < m.w1->getCols(); j++) {
            saved = m.w1->value(i, j);
            m.w1->value(i, j) += eps;
            g.w1->value(i, j) = (cost(m, ti, to) - c) / eps;
            m.w1->value(i, j) = saved;
        }
    }
    for (int i = 0; i < m.b1->getRows(); i++) {
        for (int j = 0; j < m.b1->getCols(); j++) {
            saved = m.b1->value(i, j);
            m.b1->value(i, j) += eps;
            g.b1->value(i, j) = (cost(m, ti, to) - c) / eps;
            m.b1->value(i, j) = saved;
        }
    }
    for (int i = 0; i < m.w2->getRows(); i++) {
        for (int j = 0; j < m.w2->getCols(); j++) {
            saved = m.w2->value(i, j);
            m.w2->value(i, j) += eps;
            g.w2->value(i, j) = (cost(m, ti, to) - c) / eps;
            m.w2->value(i, j) = saved;
        }
    }
    for (int i = 0; i < m.b2->getRows(); i++) {
        for (int j = 0; j < m.b2->getCols(); j++) {
            saved = m.b2->value(i, j);
            m.b2->value(i, j) += eps;
            g.b2->value(i, j) = (cost(m, ti, to) - c) / eps;
            m.b2->value(i, j) = saved;
        }
    }
}

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0};

int NSamples = sizeof(td) / sizeof(td[0]) / 3;

int main() {
    srand(time(0));
    struct Xor model;  // = xor_alloc();
    struct Xor g;      // = xor_alloc();
    // NeuralNetwork model({2, 2, 1});
    // NeuralNetwork g({2, 2, 1});
    model.w1->randomise(0, 1);
    model.b1->randomise(0, 1);
    model.w2->randomise(0, 1);
    model.b2->randomise(0, 1);
    matrix<> ti(NSamples, 2, 3, td);
    matrix<> to(NSamples, 1, 3, td + 2);

    MATRIX_PRINT(ti);
    MATRIX_PRINT(to);

    ti.print("ti", 4);

    return 0;
    float rate = 1e-1;
    float eps = 1e-1;
    // std::cout << "cost: "<<cost(model, ti,to)<<std::endl;
    for (int i = 0; i < 10000; i++) {
        std::cout << i << ": cost: " << cost(model, ti, to) << std::endl;
        finite_diff(model, g, eps, ti, to);
        xor_learn(model, g, rate);
    }
    std::cout << "cost: " << cost(model, ti, to) << std::endl;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            model.a0->value(0, 0) = i;
            model.a0->value(0, 1) = j;
            forward_xor(model);
            float y = model.a2->value(0, 0);

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
}