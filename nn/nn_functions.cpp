#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include <random>
#include <tuple>
#include "nn.h"

namespace nn {
	//-------------------------FUNCTIONS---------------------------------
	Var zeros(size_t m, size_t n) {
		return Var(std::vector<std::vector<double>>(m, std::vector<double>(n, 0.0)));
	}
	Var ones(size_t m, size_t n) {
		return Var(std::vector<std::vector<double>>(m, std::vector<double>(n, 1.0)));
	}

	Var ones_like(Var& rhs) {
		Var ans;
		ans.op = Var::ones_like;
		ans.num1 = std::make_shared<Var>(std::move(rhs));
		return ans;
	}
	Var ones_like(Var&& rhs) {
		Var ans;
		ans.op = Var::ones_like;
		ans.num1 = std::make_shared<Var>(rhs);
		return ans;
	}

	Var ones_vector(Var& rhs) {
		Var ans;
		ans.op = Var::ones_vector;
		if (rhs.graph_ptr)
			ans.num1 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num1 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var ones_vector(Var&& rhs) {
		Var ans;
		ans.op = Var::ones_vector;
		if (rhs.graph_ptr)
			ans.num1 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num1 = std::make_shared<Var>(rhs);
		return ans;
	}

	Var constant(size_t m, size_t n, double init_val) {
		Var ans(Matrix(m, n, init_val));
		ans.requires_grad = false;
		return ans;
	}
	Var shape_as(Var& rhs, double val) {
		Var ans;
		ans.op = Var::from_double;
		ans.op_num = val;
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var shape_as(Var&& rhs, double val) {
		Var ans;
		ans.op = Var::from_double;
		ans.op_num = val;
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}

	Var MSE_Loss(Var& pred, Var& label) {
		pred.requires_grad = true;
		label.requires_grad = false;

		auto total_loss = (pred - label) * (pred - label);
		return total_loss.mean();
	}

	std::vector<double> solve_linear_equation(const std::vector<std::vector<double>>& A, const std::vector<double>& B) {
		size_t m = A.size(), n = A.front().size();
		std::vector<double> X(n, 0);
		std::vector<std::vector<double>> C(m, std::vector<double>(n + 1));
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j)
				C[i][j] = A[i][j];
			C[i].back() = B[i];
		}

		//Simplify A.
		for (int j = 0;; ++j) {
			//Check the rank
			if (j == n) {
				for (int k = n; k < m; ++k) {
					if (C[k][n] != 0.0)
						throw "Rank Error!";
				}
				break;
			}

			//Swap the not 0 to the first place.
			for (int i = j;; ++i) {
				if (i == m)
					throw "Rank does not match!";
				if (C[i][j]) {
					swap(C[j], C[i]);
					break;
				}
			}

			for (int i = j + 1; i < m; ++i) {
				auto l = C[i][j] / C[j][j];
				for (int k = j; k <= n; ++k)
					C[i][k] -= C[j][k] * l;
			}
		}

		for (int i = n - 1; i >= 0; --i) {
			double tmp = C[i][n];
			for (int j = i + 1; j < n; ++j) {
				tmp -= C[i][j] * X[j];
			}
			X[i] = tmp / C[i][i];
		}

		return X;
	}

	std::tuple<std::vector<double>, double>
		linear_regression(const std::vector<std::vector<double>>& x, const std::vector<double>& y) {
		assert(x.size() == y.size());
		size_t n = x.front().size(), l = x.size();
		std::vector<double> x_mean(n, 0.0);
		double y_mean = 0.0;

		for (size_t i = 0; i < l; ++i) {
			for (size_t j = 0; j < n; ++j) {
				x_mean[j] += x[i][j] / double(l);
			}
			y_mean += y[i] / double(l);
		}

		std::vector<std::vector<double>> s(n, std::vector<double>(n, 0.0));
		std::vector<double> sy(n, 0.0);

		for (size_t j = 0; j < n; ++j) {
			for (size_t k = 0; k < n; ++k) {
				for (size_t i = 0; i < l; ++i) {
					s[j][k] += (x[i][j] - x_mean[j]) * (x[i][k] - x_mean[k]);
				}
			}
			for (size_t i = 0; i < l; ++i) {
				sy[j] += (x[i][j] - x_mean[j]) * (y[i] - y_mean);
			}
		}

		auto w = solve_linear_equation(s, sy);
		double b = y_mean;
		for (size_t i = 0; i < n; ++i)
			b -= w[i] * x_mean[i];

		return std::tuple<std::vector<double>, double>(w, b);
	}
}
