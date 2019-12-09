#include <iostream>
#include <vector>
#include <assert.h>
#include <memory>
#include <random>
#include "nn.h"

namespace nn {
	//----------------------------VAR-----------------------------------
	std::pair<int, int> Var::shape() const {
		return data.shape;
	}
	void Var::print() const {
		auto to_print = graph();
		std::cout << "Variable:(" << to_print.data.shape.first << "," << to_print.data.shape.second << ")" << std::endl;
		to_print.data.print();
	}
	std::vector<double>& Var::operator[](size_t n) {
		return data[n];
	}
	Var::~Var() {}
	Var::Var(int m, int n, bool init_random, double rand_mean, double rand_std) :data(m, n) {
		if (init_random) {
			std::default_random_engine e;
			std::normal_distribution<> n(rand_mean, rand_std);
			for (auto& p : data.data)
				for (auto& q : p)
					q = n(e);
		}
	}

	Var Var::operator=(Var& rhs) {
		op = equals;
		if (rhs.graph_ptr)
			num1 = rhs.graph_ptr;
		else
			rhs.graph_ptr = num1 = std::make_shared<Var>(rhs);
		return *this;
	}
	Var Var::operator=(Var&& rhs) {
		op = equals;
		if (rhs.graph_ptr)
			num1 = rhs.graph_ptr;
		else
			rhs.graph_ptr = num1 = std::make_shared<Var>(rhs);
		return *this;
	}
	Var Var::operator+(Var& rhs) {
		Var ans;
		ans.op = plus;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::operator+(Var&& rhs) {
		Var ans;
		ans.op = plus;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::operator-(Var& rhs) {
		Var ans;
		ans.op = minus;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::operator-(Var&& rhs) {
		Var ans;
		ans.op = minus;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::operator*(Var& rhs) {
		Var ans;
		ans.op = times;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::operator*(Var&& rhs) {
		Var ans;
		ans.op = times;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::operator/(Var& rhs) {
		Var ans;
		ans.op = devides;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::operator/(Var&& rhs) {
		Var ans;
		ans.op = devides;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::matmul(Var& rhs) {
		Var ans;
		ans.op = mm;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}
	Var Var::matmul(Var&& rhs) {
		Var ans;
		ans.op = mm;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		if (rhs.graph_ptr)
			ans.num2 = rhs.graph_ptr;
		else
			rhs.graph_ptr = ans.num2 = std::make_shared<Var>(rhs);
		return ans;
	}

	Var Var::relu() {
		Var ans;
		ans.op = re;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		return ans;
	}
	Var Var::mean() {
		Var ans;
		ans.op = means_op;
		if (graph_ptr)
			ans.num1 = graph_ptr;
		else
			graph_ptr = ans.num1 = std::make_shared<Var>(*this);
		return ans;
	}


	void Var::calculate() {
		std::unordered_set<Var*> visited;
		cal(visited);
	}
	void Var::cal(std::unordered_set<Var*>& visited) {
		if (visited.find(this) != visited.end())
			return;
		visited.insert(this);
		switch (op)
		{
		case nn::Var::none:
			break;
		case nn::Var::equals:
			num1->cal(visited);
			data = num1->data;
			break;
		case nn::Var::plus:
			num1->cal(visited);
			num2->cal(visited);
			data = num1->data + num2->data;
			break;
		case nn::Var::minus:
			num1->cal(visited);
			num2->cal(visited);
			data = num1->data - num2->data;
			break;
		case nn::Var::times:
			num1->cal(visited);
			num2->cal(visited);
			data = num1->data * num2->data;
			break;
		case nn::Var::devides:
			num1->cal(visited);
			num2->cal(visited);
			data = num1->data / num2->data;
			break;
		case nn::Var::mm:
			num1->cal(visited);
			num2->cal(visited);
			data = num1->data.matmul(num2->data);
			break;
		case nn::Var::re:
			num1->cal(visited);
			data = num1->data.relu();
			break;
		case nn::Var::from_double:
			num2->cal(visited);
			data = Matrix(num2->shape().first, num2->shape().second, op_num);
			break;
		case nn::Var::means_op: {
			num1->cal(visited);
			double mean_val = 0.0;
			for (auto p : num1->data.data)
				for (auto q : p)
					mean_val += q;
			mean_val /= (double)num1->data.shape.first * (double)num1->data.shape.second;
			data = Matrix(1, 1, mean_val);
		}
			break;
		case nn::Var::ones_like:
			num1->cal(visited);
			data = Matrix(std::vector<std::vector<double>>(num1->shape().first, std::vector<double>(num1->shape().first, 1.0)));
			break;
		case nn::Var::ones_vector:
			num1->cal(visited);
			data = Matrix(std::vector<std::vector<double>>(num1->shape().first, std::vector<double>(1, 1.0)));
			break;
		default:
			break;
		}

		//Create grad Var.
		if (requires_grad and grad.empty())
			grad = Matrix(data.shape.first, data.shape.second);
	}

	Var Var::graph() const {
		if (!graph_ptr)
			return *this;
		else
			return graph_ptr->graph();
	}

	Matrix Var::_data() {
		return graph().data;
	}

	Matrix Var::_grad() {
		return graph().grad;
	}
}
