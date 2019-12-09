#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include <random>
#include "nn.h"

namespace nn {
	void Var::zero_grad() {
		grad.clear();
		if (num1 and num1->requires_grad)
			num1->zero_grad();
		if (num2 and num2->requires_grad)
			num2->zero_grad();
	}

	void Var::backward() {
		grad = Matrix(data.shape.first, data.shape.second, 1.0);
		_backward();
	}

	void Var::_backward() {
		if (num1 and num1->requires_grad) {
			switch (op)
			{
			case nn::Var::none:
				break;
			case nn::Var::equals:
				num1->grad = grad;
				break;
			case nn::Var::plus:
				num1->grad += grad;
				break;
			case nn::Var::minus:
				num1->grad += grad;
				break;
			case nn::Var::times:
				num1->grad += num2->data * grad;
				break;
			case nn::Var::devides:
				num1->grad += grad / num2->data;
				break;
			case nn::Var::mm:
				num1->grad += grad.matmul(num2->data.transpose());
				break;
			case nn::Var::re:
				for (size_t i = 0; i < data.shape.first; ++i)
					for (size_t j = 0; j < data.shape.second; ++j)
						num1->grad.data[i][j] += (num1->data.data[i][j] > 0 ? 1 : 0)* grad[i][j];
				break;
			case nn::Var::means_op:
				num1->grad = Matrix(num1->data.shape.first, num1->data.shape.second, 1.0 / ((double)num1->data.shape.first * (double)num1->data.shape.second));
				break;
			case nn::Var::from_double:
				break;
			case nn::Var::ones_like:
				break;
			case nn::Var::ones_vector:
				break;
			default:
				break;
			}
			num1->_backward();
		}
		if (num2 and num2->requires_grad) {
			switch (op)
			{
			case nn::Var::none:
				break;
			case nn::Var::equals:
				num2->grad = grad;
				break;
			case nn::Var::plus:
				num2->grad += grad;
				break;
			case nn::Var::minus:
				num2->grad += Matrix(data.shape.first, data.shape.second, -1.0) * grad;
				break;
			case nn::Var::times:
				num2->grad += num1->data * grad;
				break;
			case nn::Var::devides:
				num2->grad -= num1->data / (num2->data * num2->data) * grad;
				break;
			case nn::Var::mm:
				num2->grad += num1->data.transpose().matmul(grad);
				break;
			case nn::Var::re:
				break;
			case nn::Var::means_op:
				break;
			case nn::Var::from_double:
				break;
			case nn::Var::ones_like:
				break;
			case nn::Var::ones_vector:
				break;
			default:
				break;
			}
			num2->_backward();
		}
	}

	void Var::optim(Optim func, double LR) {
		std::unordered_set<Var*> visited;
		switch (func)
		{
		case SGD:
			SGD_optim(LR, visited);
			break;
		case Adam:
			//Adam optimizer is not available now!
			throw "No implement now!";
			break;
		default:
			break;
		}
	}

	void Var::SGD_optim(double LR, std::unordered_set<Var*>& visited) {
		if (visited.find(this) != visited.end())
			return;
		visited.insert(this);

		if (requires_optim) {
			data -= Matrix(data.shape.first, data.shape.second, LR) * grad;
		}
		if (num1 and num1->requires_grad)
			num1->SGD_optim(LR, visited);
		if (num2 and num2->requires_grad)
			num2->SGD_optim(LR, visited);
	}
}
