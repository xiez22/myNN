# myNN
A simple nerual network framework in C++.

# Update
## 2019/12/19
- Add `solve_linear_equation` and `linear_regression` function to solve naive regression problems.
- Add `abs` function.
- Fixed some bugs.

## 2019/12/16
- __`RNN` MODULE !!!__
  It seems realy easy to add it, but I found tons of bugs when testing it. It tooks a lot of time to fix the bugs, since the stucture of the framework is really complicated.{{{(>_<)}}}>)}}}
  When using `RNN`, you should first initialize it.
  ``` C++
  auto rnn = nn::RNN();
  //Build the net.
  //......
  //Init it.
  rnn.init(BATCH_SIZE);
  
  for(size_t epoch = 0; epoch < EPOCH; ++epoch){
      //Train loop.
      //......
    
      //Call rnn.cycle() to update the hidden states.
      rnn.cycle();
  }
  ```
  __But `RNN` module has not been fully tested yet.__
- Fixed lots of bugs.
- Tested the `tanh` function and it works well.

## 2019/12/15
- Add `tanh` function and `TanH` class. But they have not been tested yet.

## 2019/12/10
- Completed the implementation of `operator=` in class `Var`.
- Improve performance of `Linear` and `Sequential` with `operator=`.
- Add the move constructor for `Var`.
- Completed the implementation of Adam Optimizer.

# How to use?
- The files are in `\nn` folder. The sample file is just in the root directory.
- Build with C++11 or higher.
- First of all, you should include `nn.h`.
- `Var` is a class with which you can easily build a caculation graph. Sample:
  ``` C++
  //The shape of matrix x is 2×3 and its val is 1.
  nn::Matrix x(2,3,1.0); 
  //The shape of a is 2×3. The shape of b is 3×2 and it is a random array.
  nn::Var a(2,3),b(3,2,true);
  auto c = a.matmul(b);

  //And now you can calculate it!
  //Set the data of Var a.
  a.data = x;
  c.calculate();

  //Calculate the grad.
  c.backward();
  //Print the grad of the a.
  a._grad().print();
  ```
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

		loss.zero_grad();
		loss.backward();
		loss.optim(nn::Var::Adam, LR);
	}
	y_.print();
  ```

## Tips & Bugs
- Adam Optimizer could be very __SLOW__ !ヽ(*。>Д<)o゜>)
- Unfinished implement of the `Tensor` class.
- __(IMPORTANT)__ Due to the restrictions of `C++`, there are some differences between `Var(const Var&)` and `Var(Var&&)`. Only values will be copied when using the former. So when you want to copy a `Var`, you are supposed to write the code like this:
  ``` C++
  nn::Var x(1,2);
  auto y = std::move(x);
  ```
  With the help of `std::move()` in C++11, you can transform a left value to a right value in order to call the move constuctor instead of copy constructor.
- The variables defined by yourself may not on the calculation graph. So we offer a function `graph()`, which returns a `Var` value that points to the node on the calculation graph. When you want to print the grad, you can write like this:
  ``` C++
  nn::Var x(init_val);
  auto y = x * x;
  y.calculate();
  y.backward();
  x._grad().print();
  ``` 
