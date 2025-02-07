# xoral_net
Neural net with inhibitory "XOR neurons"





### Setup:

1. Download
2. Create virtual environment: "python3 -m venv .venv"
3. Activate virtual environment: "source .venv/bin/activate"
4. Install requirements: "pip install -r requirements.txt"
5. Remain in virtual environment for use, or deactivate virtual environment: "deactivate"



### About:

This is a python implementation of a neural net with XOR functionality. On top of modeling a traditional neural net with layers and weights and biases, this implementation includes functionality to add layers of “XOR neurons”. The purpose of this project is to demonstrate the XOR functionality and its potential savings in artificial neural net depth and computation.

An XOR neuron, or “xoron”, is an implementation of a traditional artificial neural net neuron with an extra bias factor. It is inspired from certain human cortical neurons, which are complex enough to act as deep neural nets all in one single neuron. While there are many factors that make these specific biological neurons more computationally complex than the traditional artificial neural net neuron, this project focuses on implementing one factor: XOR functionality. Where a traditional neural net requires at least one hidden layer to solve the XOR classification problem, the biological example solves it with just one neuron. This is possible by the biological neuron inhibiting its output signal upon receiving too high an input signal.

The xoron implementation in this project achieves this behaviour by using a two-bias activation function in which as the input increases, its output increases up to a point, and then decreases back again. The first bias value translates the function horizontally, as in a traditional activation function. The second bias value “stretches” the domain of high output signal, determining the cutoff when output signal goes back from high to low. In the first and currently sole activation function implemented, this behvaiour is achieved by summing two sigmoid functions with different horizontal translations and opposite horizontal reflection, and then normalising the result. This yields the following function (s - sum of products of weights and inputs, b1 - first bias, b2 - second bias):

a( s, b1, b2 ) = [ σ( s - b1 ) + σ( -( s - b1 - b2 ) ) - 1 ] / [ 2 • σ( b2 / 2 ) - 1 ]

Apart from XOR functionality, this project implements an artificial neural net as usual, where objects of the “XoralNet” class store a list of weight matrices appropriate for each layer, a list of bias vectors, and then one more list of second bias vectors for XOR layers (or “xoral” layers). In addition to storing net parameters, the class keeps hold of input, output, and intermediate signals throughout the net. This saves on extra computation and exposes values to the user for any potential diagnostics or insight. There is also the added flexibility of specifying which layers, if any, should be xoral layers. Finally, stochastic gradient descent is used for the training function, with a constant learning rate for simplicity.

The “main.py” file explores xoral net application on a small and simple two-dimensional XOR dataset, demonstrating that while a three layer (one input, one hidden, one output layer) traditional net can solve the XOR problem, only two layers (one input, one output layer) of xoral net are needed to classify the dataset. Additionally, using the same stochastic gradient descent training process on both types of neural nets, the xoral net arrives at the same accuracy as the traditional net in less training iterations and less time. However, as a result of the constant learning rate and minimal net implementation for both nets (traditional net has layer lengths 2-2-1 and xoral net has lengths 2-1), both nets often get stuck in highly non-optimal local minima.

In conclusion, this project’s implementation of a neural net with XOR-solving neurons demonstrates depth and computation time efficiency on a small XOR dataset. Future plans include implementing more XOR activation functions, demonstrations on bigger datasets, testing more complex net architectures, and testing smarter variations in learning rate and training.


