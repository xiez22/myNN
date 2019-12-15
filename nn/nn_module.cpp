#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include <random>
#include "nn.h"

namespace nn {
	//-------------------------MODULE-------------------------------------
	Var Module::operator()(const Var& x) {
		return forward(x);
	}

	Linear::Linear(size_t in_features, size_t out_features, bool bias) :
		w(in_features, out_features, true), w_b(1, out_features, true) {
		w.requires_optim = true;
		w_b.requires_optim = true;
		m = in_features, n = out_features, if_b = bias;
	}
	Var Linear::forward(Var x) {
		w.requires_optim = true;
		auto y = x.matmul(w);
		if (if_b) {
			auto b = ones_vector(x);
			y = y + b.matmul(w_b);
		}
		return y;
	}

	Var ReLU::forward(Var x) {
		auto y = x.relu();
		return y;
	}

	Var TanH::forward(Var x) {
		auto y = x.tanh();
		return y;
	}

	Var Sequential::forward(Var x) {
		for (auto mod : seq_data)
			x = mod->operator()(x);
		return x;
	}
}
