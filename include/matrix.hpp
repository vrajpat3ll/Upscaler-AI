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
#include <fstream>
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
    unsigned m_cols = 1;
    unsigned m_rows = 1;
    unsigned m_stride = 1;
    T* m_vals;

   public:
    matrix();
    matrix(int r, int c);
    matrix(int r, int c, int s, T* v);

    /// @brief Applies the given function to all elements of the matrix.
    /// @param f The function to apply to each element of the matrix.
    void apply(T (*f)(T x));

    /// @brief `dst`.copy(`src`)
    /// @tparam T float, double, int
    /// @param src source matrix
    void copy(matrix<T> src);

    /// Fills the matrix with the specified value
    /// @param val The value to fill the matrix with.
    void fill(T val);

    unsigned getRows() const { return m_rows; }
    unsigned getCols() const { return m_cols; }

    /// @param i th row
    /// @param j th column
    /// @return reference to the value at `m_vals [ i ] [ j ]`
    T& value(unsigned i, unsigned j);
    /// @brief Don't use this function if you do not want a custom name.
    /// Instead use the MATRIX_PRINT macro.
    void print(const std::string& name, int spacing);

    void randomise(T low, T high);

    /// @brief loads the matrix from a file
    /// @param filepath path of file
    void load(const std::string& filepath);

    /// @brief saves the matrix in a file
    /// @param filepath path of file
    void save(const std::string& filepath);

    bool operator==(matrix<T>& other);
};

template <typename T>
matrix<T> mat_row(matrix<T>& m, int row) {
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
    for (unsigned i = 0; i < this->m_rows; i++) {
        for (unsigned j = 0; j < this->m_cols; j++) {
            this->value(i, j) = f(this->value(i, j));
        }
    }
}

template <typename T>
void matrix<T>::fill(T val) {
    for (unsigned i = 0; i < this->m_rows; i++) {
        for (unsigned j = 0; j < this->m_cols; j++) {
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

    for (unsigned i = 0; i < a.getRows(); i++) {
        for (unsigned j = 0; j < a.getCols(); j++) {
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

    for (unsigned i = 0; i < a.getRows(); i++) {
        for (unsigned j = 0; j < b.getCols(); j++) {
            T sum = 0;
            for (unsigned k = 0; k < a.getCols(); k++) {
                sum += a.value(i, k) * b.value(k, j);
            }
            dst.value(i, j) = sum;
        }
    }
}

template <typename T>
T& matrix<T>::value(unsigned i, unsigned j) {
    if (i >= this->m_rows) {
        fprintf(stderr, "\e[31m<Index Error> out of bound rows!\e[0m\n");
        exit(1);
    }
    if (j >= this->m_cols) {
        fprintf(stderr, "\e[31m<Index Error> out of bound cols!\e[0m\n");
        exit(1);
    }
    return ((this)->m_vals)[(i) * this->m_stride + j];
}

template <typename T>
void matrix<T>::print(const std::string& name, int spacing) {
    std::cout << std::setw(spacing) << "" << name << " = [\n";
    for (unsigned i = 0; i < this->m_rows; i++) {
        std::cout << std::setw(spacing) << "";
        for (unsigned j = 0; j < this->m_cols; j++) {
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

    for (unsigned i = 0; i < this->m_rows; i++) {
        for (unsigned j = 0; j < this->m_cols; j++) {
            this->value(i, j) = src.value(i, j);
        }
    }
}

template <typename T>
void matrix<T>::randomise(T low, T high) {
    for (unsigned i = 0; i < this->m_rows; i++) {
        for (unsigned j = 0; j < this->m_cols; j++) {
            this->value(i, j) = functions::rand_float() * (high - low) + low;
        }
    }
}

// TODO: testing remains
template <typename T>
void matrix<T>::load(const std::string& filepath) {
    // std::cout<< "LOAD METHOD UNIMPLEMENTED!\n";
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "<load> Could not open %s to save matrix.\n", filepath.c_str());
        return ;
    }
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x74616d2e && magic != 0x2e6d6174) {
        fprintf(stderr, "\e[31m<load> check data file. Improper format\n\e[0m");
        return ;
    }
    file.read(reinterpret_cast<char*>(&m_rows), 4);
    // std::cout << "<load> rows: "<< m_rows << std::endl;
    file.read(reinterpret_cast<char*>(&m_cols), 4);
    // std::cout << "<load> cols: "<< m_cols << std::endl;
    if (file.fail()) {
        throw std::runtime_error("\e[31m<load> Error reading matrix rows and columns\e[0m");
    }
    this->m_stride = m_cols;
    free(m_vals);
    this->m_vals = (T*)calloc(m_rows * m_cols, sizeof(T));
    file.read(reinterpret_cast<char*> (m_vals), m_rows*m_cols*sizeof(T));
    if (file.fail()) {
        delete[] this->m_vals;
        throw std::runtime_error("Error reading matrix data");
    }
    file.close();
}

// TODO: testing remains
template <typename T>
void matrix<T>::save(const std::string& filepath) {
    // <magic> 4bytes
    // <rows> 8 bytes
    // <cols> 8 bytes
    // <data> (rows*cols*4) bytes
    std::string magic = ".mat";
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "<save> Could not open %s to save matrix.\n", filepath.c_str());
        return ;
    }
    
    file.write(magic.c_str(), magic.size());
    file.write(reinterpret_cast<char*>(&this->m_rows), sizeof(this->m_rows));
    file.write(reinterpret_cast<char*>(&this->m_cols), sizeof(this->m_cols));
    for (unsigned i = 0; i < this->m_rows; i++) {
        file.write(reinterpret_cast<const char*>(&this->m_vals[i * this->m_stride]), this->m_cols * sizeof(T));
    }
    file.close();
}

template <typename T>
bool matrix<T>::operator==(matrix<T>& other) {
    if (this->m_rows != other.m_rows || this->m_cols != other.m_cols)
        return false;
    
    for (unsigned i = 0; i < m_rows; i++) {
        for (unsigned j = 0; j < m_cols; j++) {
            if (this->value(i, j) != other.value(i, j)) return false;
        }
    }
    return true;
}

#endif  // MATRIX_IMPLEMENTATION