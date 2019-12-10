#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include <random>
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
}
