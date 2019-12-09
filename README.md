# myNN
A simple nerual network framework in C++.

# How to use?
- The files are in `\nn` folder. The sample file is just in the root directory.
- Build with C++11 or higher.
- First of all, you should include `nn.h`.
- To define a network like this:
  ``` C++
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
  ```

- or just like this to define a sequential network simply:
  ``` C++
    auto net = nn::Sequential();
	net.add_layer(nn::Linear(1, 5));
	net.add_layer(nn::ReLU());
	net.add_layer(nn::Linear(5, 1));
  ```

- After the defination of the network, it is the computation graph. (We use static computation graph.)
    ``` C++
    nn::Var x(5, 1), y(5, 1);
	for (int i = 0; i < 5; ++i) {
		x[i][0] = i;
		y[i][0] = 3 * i * i + 2;
	}
    auto y_ = net(x);
	auto loss_func = nn::MSE_Loss;
	auto loss = loss_func(y_, y);
    ```

- Now you can train it!
  ``` C++
  for (int i = 0; i < EPOCH; ++i) {
		loss.calculate();
		loss.print();
		//y_.print();

		loss.zero_grad();
		loss.backward();
		loss.optim(nn::Var::SGD, LR);
	}
	y_.print();
  ```

## Tips & Bugs
- Unfinished implement of the `operator=`.
- Unfinished implement of the Adam Optimizer.
- Unfinished implement of the `Tensor` class.