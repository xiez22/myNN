#include <iostream>
#include "nn.h"
using namespace std;

constexpr auto EPOCH = 5000;
constexpr auto LR = 0.01;

class Net :public nn::Module {
public:
	nn::Linear fc1, fc2;
	Net() :fc1(1, 10), fc2(10, 1) {}

	nn::Var forward(nn::Var x) {
		auto y = fc1(x).relu();
		auto z = fc2(y).relu();
		return z;
	}
};

int main() {
	nn::Var x(50, 1), y(50, 1);
	for (int i = 0; i < 50; ++i) {
		x[i][0] = i;
		y[i][0] = 0.003 * i * i + 2;
	}
	x.print();
	y.print();

	auto net = nn::Sequential();
	net.add_layer(nn::Linear(1, 5));
	net.add_layer(nn::ReLU());
	net.add_layer(nn::Linear(5, 5));
	net.add_layer(nn::ReLU());
	net.add_layer(nn::Linear(5, 1));
	auto y_ = net(x);
	auto loss_func = nn::MSE_Loss;
	auto loss = loss_func(y_, y);

	for (int i = 0; i < EPOCH; ++i) {
		loss.calculate();
		if (i % 500 == 0) {
			cout << "STEP:" << i << endl;
			loss.print();
		}
		//y_.print();

		loss.zero_grad();
		loss.backward();
		loss.optim(nn::Var::Adam, LR);
	}
	y.print();
	y_.print();

	return 0;
}
