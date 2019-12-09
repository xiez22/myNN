#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include <random>
#include "nn.h"

namespace nn {
	//------------------------------MATRIX-----------------------------------
	Matrix::Matrix(const std::vector<std::vector<double>>& rhs) {
		data = rhs;
		shape.first = rhs.size();
		if (shape.first)
			shape.second = rhs[0].size();
	}

	Matrix::Matrix(size_t m, size_t n, double init_val) {
		data.resize(m, std::vector<double>(n, init_val));
		shape.first = m;
		shape.second = n;
	}

	Matrix Matrix::operator+(const Matrix& rhs) const {
		assert(rhs.shape == shape);
		Matrix ans(shape.first, shape.second);
		for (size_t i = 0; i < shape.first; ++i)
			for (size_t j = 0; j < shape.second; ++j)
				ans.data[i][j] = data[i][j] + rhs.data[i][j];
		return ans;
	}
	Matrix& Matrix::operator+=(const Matrix& rhs) {
		*this = *this + rhs;
		return *this;
	}
	Matrix Matrix::operator-(const Matrix& rhs) const {
		assert(rhs.shape == shape);
		Matrix ans(shape.first, shape.second);
		for (size_t i = 0; i < shape.first; ++i)
			for (size_t j = 0; j < shape.second; ++j)
				ans.data[i][j] = data[i][j] - rhs.data[i][j];
		return ans;
	}
	Matrix& Matrix::operator-=(const Matrix& rhs) {
		*this = *this - rhs;
		return *this;
	}
	Matrix Matrix::operator*(const Matrix& rhs) const {
		assert(rhs.shape == shape);
		Matrix ans(shape.first, shape.second);
		for (size_t i = 0; i < shape.first; ++i)
			for (size_t j = 0; j < shape.second; ++j)
				ans.data[i][j] = data[i][j] * rhs.data[i][j];
		return ans;
	}
	Matrix& Matrix::operator*=(const Matrix& rhs) {
		*this = *this * rhs;
		return *this;
	}
	Matrix Matrix::operator/(const Matrix& rhs) const {
		assert(rhs.shape == shape);
		Matrix ans(shape.first, shape.second);
		for (size_t i = 0; i < shape.first; ++i)
			for (size_t j = 0; j < shape.second; ++j)
				ans.data[i][j] = data[i][j] / rhs.data[i][j];
		return ans;
	}
	Matrix& Matrix::operator/=(const Matrix& rhs) {
		*this = *this / rhs;
		return *this;
	}
	std::vector<double>& Matrix::operator[](size_t n) {
		return data[n];
	}
	Matrix Matrix::relu() const {
		Matrix ans(shape.first, shape.second);
		for (size_t i = 0; i < shape.first; ++i) {
			for (size_t j = 0; j < shape.second; ++j) {
				ans.data[i][j] = data[i][j] > 0 ? data[i][j] : 0;
			}
		}
		return ans;
	}

	Matrix Matrix::matmul(const Matrix& rhs) const {
		assert(shape.second == rhs.shape.first);
		Matrix ans(shape.first, rhs.shape.second);
		for (size_t i = 0; i < ans.shape.first; ++i) {
			for (size_t j = 0; j < ans.shape.second; ++j) {
				for (size_t k = 0; k < shape.second; ++k) {
					ans.data[i][j] += data[i][k] * rhs.data[k][j];
				}
			}
		}
		return ans;
	}
	Matrix Matrix::transpose() const {
		Matrix ans(shape.second, shape.first);
		for (size_t i = 0; i < shape.first; ++i)
			for (size_t j = 0; j < shape.second; ++j)
				ans.data[j][i] = data[i][j];
		return ans;
	}
	void Matrix::print() const {
		std::cout << "[" << std::endl;
		for (auto p : data) {
			std::cout << "[";
			for (auto q : p) {
				std::cout << q << " ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << "]" << std::endl;
	}
	void Matrix::clear() {
		for (auto& p : data)
			for (auto& q : p)
				q = 0.0;
	}
	bool Matrix::empty() const {
		return data.empty();
	}
}
