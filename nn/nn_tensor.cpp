#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include <random>
#include "nn.h"

namespace nn {
	Tensor::Tensor(double value) :shape(std::vector<size_t>(0)), val(value), data(std::vector<Tensor>(0)) {}
	Tensor::Tensor(std::initializer_list<size_t> init_shape, double init_val) : shape(init_shape) {
		size_t dims = shape.size();
		std::vector<size_t> cur_shape{};
		Tensor cur_tensor(init_val);
		for (int i = dims - 1; i >= 0; --i) {
			std::vector<Tensor> nxt_tensor(shape[i], cur_tensor);
			cur_tensor.data = nxt_tensor;
			cur_shape.insert(cur_shape.begin(), 1, shape[i]);
			cur_tensor.shape = cur_shape;
		}
		data = cur_tensor.data;
	}

	Tensor& Tensor::operator[](size_t i) {
		return data[i];
	}

	size_t Tensor::dim() const {
		return shape.size();
	}
	void Tensor::print() const {
		std::cout << "Tensor:(";
		bool flag = true;
		for (auto p : shape) {
			if (flag)
				flag = false;
			else
				std::cout << ",";
			std::cout << p;
		}
		std::cout << ")" << std::endl << "(";
		_print();
		std::cout << ")" << std::endl;
	}

	void Tensor::_print() const {
		if (dim() == 0) {
			std::cout << val;
			return;
		}
		std::cout << "[";
		bool flag = true;
		for (auto p : data) {
			if (flag)
				flag = false;
			else {
				if (dim() == 1)
					std::cout << " ";
				else
					std::cout << std::endl;
			}
			p._print();
		}
		std::cout << "]";
	}
}
