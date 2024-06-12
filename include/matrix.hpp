#ifndef MATRIX_H
#define MATRIX_H

/*
 * This is a stb-style header file.
 * Meaning, it contains declaration as well as implementation
 * of the respective members.
 * Functionalities:
 * Addition, Apply a function, Copy, Fill, Multiplication, Print, Randomisation,
 * The `randomise` function, well fills arbitrary
 * junk values in it.
 *
 */

#include <iomanip>
#include <iostream>
#include <vector>

#define MATRIX_PRINT(mat) (mat).print(#mat, 0)

namespace functions {
float rand_float() {
    return (float)rand() / RAND_MAX;
}
}  // namespace functions

template <typename T = float>
class matrix {
   private:
    // standard stuff!
    int m_cols;
    int m_rows;
    int m_stride;
    T* m_vals;

   public:
    matrix();
    matrix(int r, int c);
    matrix(int r, int c, int s, T* v);

    /// @brief Applies the given function to all elements of the matrix.
    /// @param f The function to apply to each element of the matrix.
    void apply(T (*f)(T x));

    /// Fills the matrix with the specified value
    /// @param val The value to fill the matrix with.
    void fill(T val);
    int getRows() const { return m_rows; }
    int getCols() const { return m_cols; }

    /// @param i th row
    /// @param j th column
    /// @return reference to the value at `m_vals [ i ] [ j ]`
    T& value(int i, int j) { return ((this)->m_vals)[(i) * this->m_stride + j]; }

    /// @brief Don't use this function if you do not want a custom name.
    /// Instead use the MATRIX_PRINT macro.
    void print(const std::string& name, int spacing);

    /// @brief `dst`.copy(`src`)
    /// @tparam T float, double, int
    /// @param src source matrix
    void copy(matrix<T> src);
    void randomise(T low, T high);
};

template <typename T>
matrix<T> mat_row(const matrix<T>& m, int row) {
    return matrix<T>(1, m.getCols(), m.getCols(), &m.value(row, 0));
}
template <typename T>
void mat_dot(matrix<T>& dst, matrix<T>& a, matrix<T>& b);
template <typename T>
void mat_sum(matrix<T>& dst, matrix<T>& a, matrix<T>& b);

#endif  // MATRIX_H

#ifdef MATRIX_IMPLEMENTATION

template <typename T>
matrix<T>::matrix() : m_rows(1), m_cols(1), m_stride(1) {
    this->m_vals = (T*)malloc(m_rows * m_cols * sizeof(T));
    if (this->m_vals == NULL) {
        fprintf(stderr, "\e[31m<Constructor> Could not allocate memory!\e[0m");
        exit(1);
    }
}

template <typename T>
matrix<T>::matrix(int r, int c) : m_rows(r), m_cols(c), m_stride(c) {
    this->m_vals = (T*)calloc(m_rows * m_cols, sizeof(T));
    if (this->m_vals == NULL) {
        fprintf(stderr, "\e[31m<Constructor> Could not allocate memory!\e[0m");
        exit(1);
    }
}

template <typename T>
matrix<T>::matrix(int r, int c, int s, T* v) : m_rows(r), m_cols(c), m_stride(s), m_vals(v) {}

template <typename T>
void matrix<T>::apply(T (*f)(T x)) {
    for (int i = 0; i < this->m_rows; i++) {
        for (int j = 0; j < this->m_cols; j++) {
            this->value(i, j) = f(this->value(i, j));
        }
    }
}

template <typename T>
void matrix<T>::fill(T val) {
    for (int i = 0; i < this->m_rows; i++) {
        for (int j = 0; j < this->m_cols; j++) {
            this->value(i, j) = val;
        }
    }
}

template <typename T>
void mat_sum(matrix<T>& dst, matrix<T>& a, matrix<T>& b) {
    if (a.getRows() != b.getRows()) {
        fprintf(stderr, "\e[31m<Add function> operand_1 and operand_2's rows do not match!\e[0m\n");
        exit(1);
    }
    if (a.getCols() != b.getCols()) {
        fprintf(stderr, "\e[31m<Add function> operand_1 and operand_2's cols do not match!\e[0m\n");
        exit(1);
    }
    if (a.getRows() != dst.getRows()) {
        fprintf(stderr, "\e[31m<Add function> operand_1 and dst's rows do not match!\e[0m\n");
        exit(1);
    }
    if (a.getCols() != dst.getCols()) {
        fprintf(stderr, "\e[31m<Add function> operand_1 and dst's cols do not match!\e[0m\n");
        exit(1);
    }

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < a.getCols(); j++) {
            dst.value(i, j) = a.value(i, j) + b.value(i, j);
        }
    }
}

template <typename T>
void mat_dot(matrix<T>& dst, matrix<T>& a, matrix<T>& b) {
    if (a.getCols() != b.getRows()) {
        fprintf(stderr, "\e[31m<Multiplication function> operand_1 and operand_2's dimensions do not match!\e[0m\n");
        exit(1);
    }
    if (dst.getRows() != a.getRows()) {
        fprintf(stderr, "\e[31m<Multiplication function> operand_1 and dst's dimensions do not match!\e[0m\n");
        exit(1);
    }
    if (dst.getCols() != b.getCols()) {
        fprintf(stderr, "\e[31m<Multiplication function> operand_2 and dst's dimensions do not match!\e[0m\n");
        exit(1);
    }

    for (int i = 0; i < a.getRows(); i++) {
        for (int j = 0; j < b.getCols(); j++) {
            dst.value(i, j) = 0;
            for (int k = 0; k < a.getCols(); k++) {
                dst.value(i, j) += a.value(i, k) * b.value(k, j);
            }
        }
    }
}

template <typename T>
void matrix<T>::print(const std::string& name, int spacing) {
    std::cout << std::setw(spacing) << "" << name << " = [\n";
    for (int i = 0; i < this->m_rows; i++) {
        std::cout << std::setw(spacing) << "";
        for (int j = 0; j < this->m_cols; j++) {
            std::cout << "    " << std::setprecision(6) << this->value(i, j) << ',';
        }
        std::cout << '\n';
    }
    std::cout << std::setw(spacing) << "" << "]\n";
}

template <typename T>
void matrix<T>::copy(matrix<T> src) {
    if (this->m_rows != src.getRows()) {
        fprintf(stderr, "\e[31m<Copy function> Rows do not match!\e[0m");
        exit(1);
    }
    if (this->m_cols != src.getCols()) {
        fprintf(stderr, "\e[31m<Copy function> Cols do not match!\e[0m");
        exit(1);
    }

    for (int i = 0; i < this->m_rows; i++) {
        for (int j = 0; j < this->m_cols; j++) {
            this->value(i, j) = src.value(i, j);
        }
    }
}

template <typename T>
void matrix<T>::randomise(T low, T high) {
    for (int i = 0; i < this->m_rows; i++) {
        for (int j = 0; j < this->m_cols; j++) {
            this->value(i, j) = functions::rand_float() * (high - low) + low;
        }
    }
}

#endif  // MATRIX_IMPLEMENTATION