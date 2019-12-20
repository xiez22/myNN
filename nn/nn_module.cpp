#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <random>
#include <tuple>
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
	
	RNNCell::RNNCell(size_t in_features, size_t out_features, bool bias, bool nonlinearity) :
		wih(in_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		whh(out_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_b(1, out_features, true, 0.0, 1.0 / sqrt(out_features)) {
		wih.requires_optim = true, whh.requires_optim = true, w_b.requires_optim = true;
		m = in_features, n = out_features, if_b = bias, if_tanh = nonlinearity;
	}

	void LSTM::init(size_t batch_size) {
		h_s_tmp.set_data(Matrix(batch_size, n));
		c_s_tmp.set_data(Matrix(batch_size, n));
	}

	Var RNNCell::forward(Var& x) {
		h_states = x.matmul(wih) + h_states.matmul(whh);
		if (if_b) {
			auto b = ones_vector(x);
			h_states = h_states + b.matmul(w_b);
		}
		if (if_tanh)
			h_states = h_states.tanh();
		else
			h_states = h_states.relu();
		return h_states;
	}

	std::vector<Var> RNN::operator()(std::vector<Var>& x) {
		std::vector<Var> y;
		rnn_cell.h_states = h_s_in;
		for (auto& p : x)
			y.emplace_back(rnn_cell(p));
		h_s_out = rnn_cell.h_states;
		return y;
	}

	void RNN::init(size_t batch_size) {
		h_s_in.set_data(Matrix(batch_size, n));
	}

	void RNN::cycle() {
		h_s_in.set_data(h_s_out);
	}

	Var LSTM::forward(Var& x) {
		auto i_t = x.matmul(w_ii) + h_s_tmp.matmul(w_hi);
		auto f_t = x.matmul(w_if) + h_s_tmp.matmul(w_hf);
		auto g_t = x.matmul(w_ig) + h_s_tmp.matmul(w_hg);
		auto o_t = x.matmul(w_io) + h_s_tmp.matmul(w_ho);

		if (if_b) {
			auto b = ones_vector(x);
			i_t = i_t + b.matmul(b_i);
			f_t = f_t + b.matmul(b_f);
			g_t = g_t + b.matmul(b_g);
			o_t = o_t + b.matmul(b_o);
		}

		i_t = i_t.sigmoid();
		f_t = f_t.sigmoid();
		g_t = g_t.tanh();
		o_t = o_t.sigmoid();

		auto c_t = f_t * c_s_tmp + i_t * g_t;
		auto h_t = o_t * c_t.tanh();
		h_s = h_t.copy(), c_s = c_t.copy();
		return h_t;
	}

	void LSTM::cycle() {
		h_s.calculate();
		c_s.calculate();
		h_s_tmp.set_data(h_s);
		c_s_tmp.set_data(c_s);
	}

	LSTM::LSTM(size_t in_features, size_t out_features, bool bias) :
		w_ii(in_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_if(in_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_ig(in_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_io(in_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_hi(out_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_hf(out_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_hg(out_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		w_ho(out_features, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		b_i(1, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		b_f(1, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		b_g(1, out_features, true, 0.0, 1.0 / sqrt(out_features)),
		b_o(1, out_features, true, 0.0, 1.0 / sqrt(out_features)) {
		w_ii.requires_optim = true, w_if.requires_optim = true;
		w_ig.requires_optim = true, w_io.requires_optim = true;
		w_hi.requires_optim = true, w_hf.requires_optim = true;
		w_hg.requires_optim = true, w_io.requires_optim = true;
		b_i.requires_optim = true, b_f.requires_optim = true;
		b_g.requires_optim = true, b_o.requires_optim = true;
		h_s.requires_grad = h_s_tmp.requires_grad = false;
		c_s.requires_grad = c_s_tmp.requires_grad = false;
		m = in_features, n = out_features, if_b = bias;
	}

	Var ReLU::forward(Var& x) {
		auto y = x.relu();
		return y;
	}

	Var TanH::forward(Var& x) {
		auto y = x.tanh();
		return y;
	}

	Var Sigmoid::forward(Var& x) {
		auto y = x.sigmoid();
		return y;
	}

	Var Sequential::forward(Var& x) {
		auto y = std::move(x);
		for (auto mod : seq_data)
			y = mod->operator()(y);
		return y;
	}
}
