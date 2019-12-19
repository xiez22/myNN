#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <random>
#include "nn.h"

namespace nn {
	//-------------------------MODULE-------------------------------------
	Var Module::operator()(Var& x) {
		return forward(x);
	}
	Var Module::operator()(Var&& x) {
		return forward(x);
	}

	Linear::Linear(size_t in_features, size_t out_features, bool bias) :
		w(in_features, out_features, true), w_b(1, out_features, true) {
		w.requires_optim = true;
		w_b.requires_optim = true;
		m = in_features, n = out_features, if_b = bias;
	}
	Var Linear::forward(Var& x) {
		auto y = x.matmul(w);
		if (if_b) {
			auto b = ones_vector(x);
			y = y + b.matmul(w_b);
		}
		return y;
	}
	
	RNN::RNN(size_t in_features, size_t out_features, bool bias, bool nonlinearity) :
		wih(in_features, out_features, true), whh(out_features, out_features, true),
		w_b(1, out_features, true) {
		wih.requires_optim = true, whh.requires_optim = true, w_b.requires_optim = true;
		h_states.requires_grad = false;
		m = in_features, n = out_features, if_b = bias;
	}

	void RNN::init(size_t batch_size) {
		if (h_states_tmp.graph_ptr)
			h_states_tmp.graph_ptr->data = Matrix(batch_size, n);
		else
			h_states_tmp.data = Matrix(batch_size, n);
	}

	Var RNN::forward(Var& x) {
		auto y = x.matmul(wih);
		y = y + h_states_tmp.matmul(whh);
		if (if_b) {
			auto b = ones_vector(x);
			y = y + b.matmul(w_b);
		}
		if (if_tanh)
			y = y.tanh();
		else
			y = y.relu();
		h_states = y.copy();
		return y;
	}

	void RNN::cycle() {
		h_states.calculate();
		if (h_states_tmp.graph_ptr)
			h_states_tmp.graph_ptr->data = h_states._data();
		else
			h_states_tmp.data = h_states._data();
	}

	Var ReLU::forward(Var& x) {
		auto y = x.relu();
		return y;
	}

	Var TanH::forward(Var& x) {
		auto y = x.tanh();
		return y;
	}

	Var Sequential::forward(Var& x) {
		auto y = std::move(x);
		for (auto mod : seq_data)
			y = mod->operator()(y);
		return y;
	}
}
