#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_set>
#include <tuple>

namespace nn {
	//A simple matrix class to implement basic matrix operations.
	class Matrix {
	public:
		Matrix() = default;
		Matrix(const std::vector<std::vector<double>>&);
		Matrix(size_t m, size_t n, double init_val = 0.0);
		std::pair<size_t, size_t> shape;

		std::vector<std::vector<double>> data;

		Matrix operator+(const Matrix& rhs) const;
		Matrix& operator+=(const Matrix& rhs);
		Matrix operator-(const Matrix& rhs) const;
		Matrix& operator-=(const Matrix& rhs);
		Matrix operator*(const Matrix& rhs) const;
		Matrix& operator*=(const Matrix& rhs);
		Matrix operator/(const Matrix& rhs) const;
		Matrix& operator/=(const Matrix& rhs);
		Matrix relu() const;
		std::vector<double>& operator[](size_t n);

		Matrix matmul(const Matrix& rhs) const;
		Matrix transpose() const;
		void print() const;
		void clear();
		bool empty() const;
	};

	//A Var class that includes some basic NN functions.
	class Var {
	public:
		enum Var_op { none, equals, plus, minus, times, devides, mm, re, th, ab, from_double, ones_like, ones_vector, means_op };
		enum Optim { SGD, Adam };
		//Adam Optimizer Parameters.
		Matrix adam_m, adam_v;
		int adam_t = 0;

		//Graph_ptr is a pointer that points to the real Var on the calculation graph.
		std::shared_ptr<Var> num1 = nullptr, num2 = nullptr, graph_ptr = nullptr;
		Matrix data, grad;
		Var_op op = Var_op::none;
		bool requires_grad = true, requires_optim = false;
		double op_num = 0.0;

		Var() = default;
		Var(const Matrix& matrix) :data(matrix) {}
		Var(Var&& rhs);
		Var(const Var&) = default;
		Var(const std::vector<std::vector<double>>& v) :data(v) {}
		Var(int m, int n, bool init_random = false, double rand_mean = 0.0, double rand_std = 1.0);
		~Var();

		std::pair<size_t, size_t> shape() const;
		void print() const;
		Var graph() const;
		Matrix _data();
		Matrix _grad();
		std::vector<double>& operator[](size_t n);
		bool empty() const;
		Var copy();

		Var operator=(Var& rhs);
		Var operator=(Var&& rhs);
		Var operator+(Var& rhs);
		Var operator+(Var&& rhs);
		Var operator-(Var& rhs);
		Var operator-(Var&& rhs);
		Var operator*(Var& rhs);
		Var operator*(Var&& rhs);
		Var operator/(Var& rhs);
		Var operator/(Var&& rhs);
		Var matmul(Var& rhs);
		Var matmul(Var&& rhs);
		Var relu();
		Var tanh();
		Var mean();
		Var abs();

		void calculate();
		void zero_grad();
		void backward();
		void optim(Optim func = SGD, double LR = 0.001);
	protected:
		void cal(std::unordered_set<Var*>&);
		void _backward();
		void SGD_optim(double, std::unordered_set<Var*>&);
		void Adam_optim(double, double, double, std::unordered_set<Var*>&);
	};

	//A Scalar class for Tensor.
	class Scalar {
	public:
		enum fn_op { none, equals, plus, minus, times, devides, relu };
		double val = 0.0;
		fn_op fn = none;
	};

	//A Tensor class for other use.
	//(Maybe it will replace the Var in the future.)
	class Tensor {
	public:
		std::vector<Tensor> data;
		std::vector<size_t> shape;
		double val = 0.0;

		Tensor() = default;
		Tensor(double value);
		Tensor(std::initializer_list<size_t> init_shape, double init_val = 0.0);

		Tensor& operator[](size_t i);
		size_t dim() const;
		void print() const;

	protected:
		void _print() const;
	};

	//NN module.
	class Module {
		Var in_layer;
	public:
		Module() = default;
		Var operator()(Var&);
		Var operator()(Var&&);

		virtual Var forward(Var&) = 0;
	};

	class Linear :public Module {
		size_t m = 0, n = 0;
		bool if_b = true;
		Var w, w_b;
	public:
		Linear(size_t in_features, size_t out_features, bool bias = true);
		Var forward(Var&);
	};

	class RNN :public Module {
		size_t m = 0, n = 0;
		bool if_b = true, if_tanh = true;
		Var wih, whh, w_b;
	public:
		//A var to storage the hidden states.
		Var h_states, h_states_tmp;
		RNN(size_t in_features, size_t out_features,
			bool bias = true, bool nonlinearity = true);
		//Clear the hidden states.
		void init(size_t batch_size);
		void cycle();
		Var forward(Var&) override;
	};

	class ReLU :public Module {
	public:
		ReLU() = default;
		Var forward(Var&);
	};

	class TanH :public Module {
	public:
		TanH() = default;
		Var forward(Var&);
	};

	class Sequential :public Module {
		std::vector<std::shared_ptr<Module>> seq_data;
	public:
		Sequential() = default;
		template<class Module_Type>
		void add_layer(const Module_Type& layer) {
			seq_data.emplace_back(std::make_shared<Module_Type>(layer));
		}

		Var forward(Var&);
	};

	

	//-------------------Functions--------------------------
	Var zeros(size_t m, size_t n);
	Var ones(size_t m, size_t n);
	Var constant(size_t m, size_t n, double init_val);
	Var ones_like(Var&);
	Var ones_like(Var&&);
	Var ones_vector(Var&);
	Var ones_vector(Var&);
	Var shape_as(Var&, double val = 0.0);
	Var shape_as(Var&&, double val = 0.0);

	Var MSE_Loss(Var& pred, Var& label);
	
	//Input:X,Y Output:W,B
	std::vector<double> solve_linear_equation(const std::vector<std::vector<double>>& A, const std::vector<double>& B);
	std::tuple<std::vector<double>, double>
		linear_regression(const std::vector<std::vector<double>>&, const std::vector<double>&);
}
