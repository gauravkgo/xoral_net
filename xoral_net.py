
import numpy as np
from scipy.special import expit



class XoralNet:

  """
  This class is an implementation of a neural net, with the ability to use XOR neurons for one or more of layers as opposed to traditional neurons.
  XOR neurons, or xorons, have two bias values instead of the one. While one bias value determines when a signal is high enough to trigger high output, the second bias determines a second cutoff where the signal is too high and triggers low output again.

  Notes:
  - Several attributes and functions have "batch" in their name. This refers to holding / processing multiple points of data in parallel, usually in the case of processing multiple input samples in a batch at once.
  - Net layers are indexed such that the input layer is first and the output layer is last. Internally, these are often indexed with negative indices to consistently index across lists with lengths differing by one (such as neuron layers vs bias layers).

  Attributes:
  - neural_activation_lookup (dictionary): Lookup table for traditional activation functions and their respective partial derivatives.
  - xoral_activation_lookup (dictionary): Lookup table for XOR neuron activation functions and their respective partial derivatives.
  - cost_lookup (dictionary): Lookup table for cost functions and their respective partial derivatives.
  - layer_lengths (list of ints): The lengths of each layer of the net (number of neurons in each layer). List size is the number of layers in the net.
  - num_layers (int): The number of layers in the net (depth).
  - layer_is_xor (list of booleans): Describes whether a layer has traditional or XOR neurons. If the element at index i is True, then layer i is an XOR layer. The first / 0'th input layer always never has weights nor biases. List size is the number of layers in the net.
  - neuron_layers_batch (list of numpy ndarrays): Contains the values/signals stored in each neuron. Has dimensions l by n by i. l is the number of layers. n is the length of the parallel batch (i.e. how many samples running in parallel). i is the length of a specific layer.
  - weighted_sum_layers_batch (list of numpy ndarrays): Temporarily stores the sums of signal-weight products for use in backpropagation for training. Has dimensions l-1 by n by i. l is the number of layers. n is the length of the parallel batch (i.e. how many samples running in parallel). i is the length of a specific layer.
  - weight_layers (list of numpy ndarrays): Contains the weights of each neuron in each layer as a list of matrices. List length is the number of layers minus one. Each element in the list is the weight matrix of that index's layer. The matrix has dimensions i by j, where i is the length of the specific neuron layer, and j is the length of the prior layer.
  - bias1_layers (list of numpy ndarrays): Contains the first set of biases of each neuron in each layer as a list of vectors. List length is the number of layers minus one. Each element in the list is the first bias vector of that index's layer. The vector length is the length of the layer. If the layer is an XOR layer, this is the first of two sets of biases, otherwise it is the sole bias vector.
  - bias2_layers (list of numpy ndarrays): Contains the second set of biases of each neuron in each layer as a list of vectors. List length is the number of layers minus one. Each element in the list is the second bias vector of that index's layer. The vector length is the length of the layer. Second biases are only used for XOR layers. If a layer is a traditional neural layer, the respective second bias vector is None.
  - activation_types (list of strings): List of activation types for each layer in the net, excluding the input layer. The length of the list is the number of layers minus one. Possible activation types include: sigmoid.
  - activation_function_layers (list of functions): List of callable activation functions for each layer in the net, excluding the input layer. The length of the list is the number of layers minus one. Functions must handle vectorisation / parallelisation with numpy ndarray arguments.
  - d_activation_d_sum_layers (list of functions): List of callable functions for each layer in the net, excluding the input layer, representing partial derivatives of activation with respect to weighted sum. The length of the list is the number of layers minus one. Functions must handle vectorisation / parallelisation with numpy ndarray arguments.
  - d_activation_d_bias1_layers (list of functions): List of callable functions for each layer in the net, excluding the input layer, representing partial derivatives of activation with respect to first set of bias. The length of the list is the number of layers minus one. Functions must handle vectorisation / parallelisation with numpy ndarray arguments.
  - d_activation_d_bias2_layers (list of functions): List of callable functions for each layer in the net, excluding the input layer, representing partial derivatives of activation with respect to second set of bias. The length of the list is the number of layers minus one. Functions must handle vectorisation / parallelisation with numpy ndarray arguments. For non XOR layers, the corresponding list element is None.
  - cost_type (string): Type of cost function to be used in calculating cost and cost gradient for training. Possible types include: least_squares.
  - cost_function (function): Callable cost function to be used in calculating cost / loss. Must be able to handle vectorisation / parallelisation at the level of calculating cost from one expected vector and one predicted vector at once. Does not need to handle batch parallelisation.
  - d_cost_d_output (function): Callable function representing the partial derivative of cost with respect to output layer, for calculating cost gradient for training. Must be able to handle vectorisation / parallelisation at the level of calculating cost derivative from one expected vector and one predicted vector at once. Does not need to handle batch parallelisation.

  Methods:
  - __init__: Initialisation of neural net.
  - run: Run one input sample vector through the net and compute resulting predicted output.
  - run_batch: Run a batch of input sample vectors, compute resulting predicted batch output.
  - vector_to_class: Convert an output vector into an integer class label.
  - vectors_to_classes: Convert multiple output vectors into integer class labels.
  - get_accuracy: Calculate accuracy of the net following a run / run batch operation.
  - get_cost_batch_average: Calculate average cost of the net following a run / run batch operation.
  - get_cost_gradient_batch_average: Calculate average cost gradient following a run / run batch operation, for training purposes.
  - train: Train the neural net to learn / improve its prediction.
  """



  neural_activation_lookup = {

    # 'd' in variable names means partial derivative
    # 's' in expressions means sum of inputs x weights
    # 'b1' in expressions means bias1

    "sigmoid": {
      "activation": lambda s, b1: expit(s + b1),
      "d_activation_d_sum": lambda s, b1: expit(s + b1) * (1 - expit(s + b1)),
      "d_activation_d_bias1": lambda s, b1: expit(s + b1) * (1 - expit(s + b1)),
    },
  }

  xoral_activation_lookup = {

    # 'd' in variable names means partial derivative
    # 's' in expressions means sum of inputs x weights
    # 'b1' in expressions means bias1
    # 'b2' in expressions means bias2

    "sigmoid": {
      "activation": lambda s, b1, b2: (expit(s - b1) + expit(-(s - b1 - b2)) - 1) / (2 * expit(b2 / 2) - 1),
      "d_activation_d_sum": lambda s, b1, b2: (expit(s - b1) * (1 - expit(s - b1)) - expit(-(s - b1 - b2)) * (1 - expit(-(s - b1 - b2)))) / (2 * expit(b2 / 2) - 1),
      "d_activation_d_bias1": lambda s, b1, b2: (expit(-(s - b1 - b2)) * (1 - expit(-(s - b1 - b2))) - expit(s - b1) * (1 - expit(s - b1))) / (2 * expit(b2 / 2) - 1),
      "d_activation_d_bias2": lambda s, b1, b2: expit(-(s - b1 - b2)) * (1 - expit(-(s - b1 - b2))) / (2 * expit(b2 / 2) - 1),
    },
  }

  cost_lookup = {

    # 'd' in variable names means partial derivative
    # 'o' in expressions means predicted output values
    # 'y' in expressions means expected output values

    "least_squares": {
      "cost": lambda o, y: np.sum((o - y) ** 2),
      "d_cost_d_output": lambda o, y: 2 * (o - y),
    },
  }



  def __init__(self,
      layer_lengths: list[int],
      activation_types: str | list[str] = "sigmoid",
      xor_layers: list[int] = [],
      cost_type: str = "least_squares"):
    
    """
    Initialise a neural/xoral net and specify activation type and neuron type for each layer.

    Parameters:
    - layer_lengths (list of ints): Number of neurons in each layer. List length is the number of layers.
    - activation_types (string or list of strings): Activation function type for each layer excluding input layer. Specifying one type sets that activation for all layers. Possible types are: sigmoid.
    - xor_layers (list of ints): List of indices of layers which have XOR type neurons rather than traditional neurons. An empty list sets all layers to traditional neuron layers.
    - cost_type (string): Type of cost function to train the net with. Possible types are: least-squares.
    """

    self.layer_lengths = layer_lengths
    self.num_layers = len(layer_lengths)
    self.layer_is_xor = [i in xor_layers for i in range(self.num_layers)]
    self.neuron_layers_batch = [np.zeros((1, layer_len)) for layer_len in layer_lengths]
    self.weighted_sum_layers_batch = [np.zeros((1, layer_len)) for layer_len in layer_lengths]
    self.weight_layers = [np.random.randn(*layer_lengths[i : i + 2][ : : -1]) for i in range(self.num_layers - 1)]
    self.bias1_layers = [np.random.randn(layer_lengths[i]) for i in range(1, self.num_layers)]
    self.bias2_layers = [np.random.randn(layer_lengths[i]) if self.layer_is_xor[i] else None for i in range(1, self.num_layers)]

    self.activation_types = activation_types if type(activation_types) is list else [activation_types for _ in range(1, self.num_layers)]
    function_groups = [self.neural_activation_lookup[act_type] if not self.layer_is_xor[i + 1] else self.xoral_activation_lookup[act_type] for i, act_type in enumerate(self.activation_types)]
    self.activation_function_layers = [functions["activation"] for functions in function_groups]
    self.d_activation_d_sum_layers = [functions["d_activation_d_sum"] for functions in function_groups]
    self.d_activation_d_bias1_layers = [functions["d_activation_d_bias1"] for functions in function_groups]
    self.d_activation_d_bias2_layers = [functions["d_activation_d_bias2"] if "d_activation_d_bias2" in functions.keys() else None for functions in function_groups]

    self.cost_type = cost_type
    functions = self.cost_lookup[cost_type]
    self.cost_function = functions["cost"]
    self.d_cost_d_output = functions["d_cost_d_output"]



  def run(self,
      x: np.ndarray) -> np.ndarray:
    
    """
    Compute net output from a single input vector.

    Parameters:
    - x (numpy ndarray): Input vector with the same length as the first layer of the net.

    Returns:
    - output layer (numpy ndarray): Predicted output of net resulting from input x.
    """

    return self.run_batch(x[np.newaxis, :])[0]

  def run_batch(self,
      X: np.ndarray) -> np.ndarray:
    
    """
    Compute net outputs from multiple input vectors in parallel.

    Parameters:
    - X (numpy ndarray): Input vectors stacked into an array of dimensions n by l, where n is the number of input vectors and l is the length of each vector. l must be same length as the first layer of the net.

    Returns:
    - output layer batch (numpy ndarray): Predicted outputs resulting from X, stacked into an array of dimensions n by l, where n is the number of output vectors and l is the length of each output vector.
    """

    # dims:
    # n = number of input samples/runs being processed in parallel
    # i = length of respective layer
    # j = length of the previous layer

    self.neuron_layers_batch[0] = X.reshape(X.shape[0], -1) # dims (n, input layer len)
    # use negative indices for consistent indexing through lists with lengths differing by 1
    for layer_i in range(-1, -self.num_layers, -1)[ : : -1]:
      previous_actions_layer_batch = self.neuron_layers_batch[layer_i - 1] # dims (n, j)
      current_sums_layer_batch = previous_actions_layer_batch @ self.weight_layers[layer_i].T # dims (n, i)
      self.weighted_sum_layers_batch[layer_i] = current_sums_layer_batch # dims (n, i)
      activation = self.activation_function_layers[layer_i]
      if self.layer_is_xor[layer_i]: current_actions_layer_batch = activation(current_sums_layer_batch, self.bias1_layers[layer_i], self.bias2_layers[layer_i]) # dims (n, i)
      else: current_actions_layer_batch = activation(current_sums_layer_batch, self.bias1_layers[layer_i]) # dims (n, i)
      self.neuron_layers_batch[layer_i] = current_actions_layer_batch # dims (n, i)
    return self.neuron_layers_batch[-1] # dims (n, output layer len)



  def vector_to_class(self,
      y: np.ndarray) -> int:
    
    """
    Translate an output vector to an integer class label. If the vector is one element long, class label is 0 or 1 depending on which value is closer.

    Parameters:
    - y (numpy ndarray): Output vector to translate into class label. Must be same length as net's output layer.
    
    Returns:
    - class label (int): Translated class label. If y vector length is 1, label is 0 or 1 depending on which value is closer.
    """

    return self.vectors_to_classes(y[np.newaxis, :])[0]

  def vectors_to_classes(self,
      Y: np.ndarray) -> np.ndarray:
    
    """
    Translate output vectors to integer class labels. If each vector is one element long, class label is 0 or 1 depending on which value is closer.

    Parameters:
    - Y (numpy ndarray): Batch of output vector to translate into class labels. Dimensions are n by i, where n is the number of vectors in the batch, and i is the length of each vector. i must be equal to the length of the net's output layer.
    
    Returns:
    - class labels (numpy ndarray of ints): Translated class labels. Vector length is the number of passed in output vectors in the batch. If an output vector length is 1, label is 0 or 1 depending on which value is closer.
    """

    return np.argmax(Y, axis = 1) if len(Y[0]) > 1 else np.where(Y[:, 0] < 0.5, np.zeros(Y.shape[0]), np.ones(Y.shape[0]))



  def get_accuracy(self,
      Y_classes: np.ndarray) -> float:
    
    """
    Calculate classification accuracy of net following a run/run_batch operation.

    Parameters:
    - Y_classes (numpy ndarray): Correct, expected integer class labels with which to compare net's results with. An int vector who's length is the same as the number of samples passed in the prior run operation. Output vectors can be turned into int class labels with vector_to_class/vectors_to_classes operation.

    Returns:
    - accuracy (float): Accuracy of how many times predicted class labels match correct class labels, as a decimal from 0 to 1.
    """

    Y_predicted = self.neuron_layers_batch[-1]
    Y_predicted_classes = self.vectors_to_classes(Y_predicted)
    accuracy = (Y_predicted_classes == Y_classes).sum() / len(Y_classes)
    return accuracy



  def get_cost_batch_average(self,
      Y: np.ndarray) -> float:
    
    """
    Calculate batch average cost of the net following a run/run_batch operation.

    Parameters:
    - Y (numpy ndarray): Correct, expected output vectors with which to compare net's results with for cost computation. Dimensions must be n by i, where n is the number of samples passed in the prior run operation, and i is the length of the net outer layer.

    Returns:
    - average cost (float): Batch average cost calculated from comparing predicted and expected outputs, using the cost function specified during net initialisation.
    """

    Y_predicted = self.neuron_layers_batch[-1]
    batch_cost = np.array([self.cost_function(Y_predicted[i], Y[i]) for i in range(len(Y))])
    average_cost = batch_cost.mean()
    return average_cost



  def get_cost_gradient_batch_average(self,
      Y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
    
    """
    Calculate the batch average of cost gradients following a run/run_batch operation using backpropagation.

    Parameters:
    - Y (numpy ndarray): Correct, expected output vectors with which to compare net's results with for gradient computation. Dimensions must be n by i, where n is the number of samples passed in the prior run operation, and i is the length of the net outer layer.

    Returns:
    - average cost gradient on weight layers (list of numpy ndarrays): Batch average gradient on each layer's weight matrices, organised into a list of arrays. The element at index 'i' is the batch average gradient on the weight matrix for layer 'i'. Each array is organised into dimensions n by i by j, where n is the batch length and i and j index through a weight matrix as usual.
    - average cost gradient on bias_1 layers (list of numpy ndarrays): Batch average gradient on each layer's first bias vector, organised into a list of vectors. The element at index 'i' is the batch average gradient on the bias vector for layer 'i'. If a layer is an XOR layer, these are the gradient on the first set of biases, otherwise the gradient on the sole set of biases. Each gradient vector has same length as the first bias vector at that layer.
    - average cost gradient on bias_2 layers (list of numpy ndarrays): Batch average gradient on each layer's second bias vector, organised into a list of vectors. The element at index 'i' is the batch average gradient on the second bias vector for layer 'i'. If a layer is not an XOR layer, the corresponding list element is None. Each gradient vector has the same length as the second bias vector at that layer.
    """

    # dims:
    # l = number of weight/bias layers (number of neuron layers - 1)
    # n = batch size (number of samples processed in parallel)
    # i = number of neurons in current layer
    # j = number of neurons in the prior layer

    cost_gradient_on_weight_layers_batch = [] # (l, n, i, j)
    cost_gradient_on_bias1_layers_batch = [] # (l, n, i)
    cost_gradient_on_bias2_layers_batch = [] # (l, n, i)

    d_cost_d_current_actions = np.array([self.d_cost_d_output(self.neuron_layers_batch[-1][i], Y[i]) for i in range(len(Y))])

    # Negative indexing for consistent indexing through lists differing in length by 1
    for layer_i in range(-1, -len(self.layer_lengths), -1):

      previous_actions_batch = self.neuron_layers_batch[layer_i - 1] # (n, j)
      sums_batch = self.weighted_sum_layers_batch[layer_i] # (n, i)
      weights = self.weight_layers[layer_i] # (i, j)
      bias1_vector = self.bias1_layers[layer_i] # (n, i)
      bias2_vector = self.bias2_layers[layer_i] # (n, i)

      d_sums_d_weights_batch = previous_actions_batch # (n, j)
      d_sums_d_previous_actions = weights # (i, j)
      d_activation_d_sum = self.d_activation_d_sum_layers[layer_i]
      d_activation_d_bias1 = self.d_activation_d_bias1_layers[layer_i]
      d_activation_d_bias2 = self.d_activation_d_bias2_layers[layer_i]
      if self.layer_is_xor[layer_i]:
        d_actions_d_sums_batch = d_activation_d_sum(sums_batch, bias1_vector, bias2_vector) # (n, i)
        d_actions_d_bias1_batch = d_activation_d_bias1(sums_batch, bias1_vector, bias2_vector) # (n, i)
        d_actions_d_bias2_batch = d_activation_d_bias2(sums_batch, bias1_vector, bias2_vector) # (n, i)
      else:
        d_actions_d_sums_batch = d_activation_d_sum(sums_batch, bias1_vector) # (n, i)
        d_actions_d_bias1_batch = d_activation_d_bias1(sums_batch, bias1_vector) # (n, i)
        d_actions_d_bias2_batch = None
      
      d_cost_d_weights_batch = (d_cost_d_current_actions * d_actions_d_sums_batch)[:, :, np.newaxis] @ d_sums_d_weights_batch[:, np.newaxis, :] # (n, i, j)
      d_cost_d_bias1_batch = d_cost_d_current_actions * d_actions_d_bias1_batch # (n, i)
      d_cost_d_bias2_batch = d_cost_d_current_actions * d_actions_d_bias2_batch if self.layer_is_xor[layer_i] else None # (n, i)
      d_cost_d_previous_actions = np.sum((d_cost_d_current_actions * d_actions_d_sums_batch)[:, :, np.newaxis] * d_sums_d_previous_actions, axis = 1) # (n, j)

      cost_gradient_on_weight_layers_batch.insert(0, d_cost_d_weights_batch)
      cost_gradient_on_bias1_layers_batch.insert(0, d_cost_d_bias1_batch)
      cost_gradient_on_bias2_layers_batch.insert(0, d_cost_d_bias2_batch)
      d_cost_d_current_actions = d_cost_d_previous_actions

    average_cost_gradient_on_weight_layers = [batch_cost_gradient_on_weight_layer.mean(axis = 0) for batch_cost_gradient_on_weight_layer in cost_gradient_on_weight_layers_batch]
    average_cost_gradient_on_bias1_layers = [batch_cost_gradient_on_bias1_layer.mean(axis = 0) for batch_cost_gradient_on_bias1_layer in cost_gradient_on_bias1_layers_batch]
    average_cost_gradient_on_bias2_layers = [batch_cost_gradient_on_bias2_layer.mean(axis = 0) if batch_cost_gradient_on_bias2_layer is not None else None for batch_cost_gradient_on_bias2_layer in cost_gradient_on_bias2_layers_batch]

    return average_cost_gradient_on_weight_layers, average_cost_gradient_on_bias1_layers, average_cost_gradient_on_bias2_layers



  def train(self,
      X: np.ndarray,
      Y: np.ndarray,
      mini_batch_size: int,
      learning_rate: float = 0.01,
      iterations: int = 1):
    
    """
    Train the net given training data and hyperparameters. Shuffles training data and uses stochastic gradient descent method.

    Parameters:
    - X (numpy ndarray): Training sample input vectors stacked into an array of dimensions n by l, where n is the number of input vectors and l is the length of each vector. l must be same length as the first layer of the net.
    - Y (numpy ndarray): Expected output vectors corresponding to the training input vectors, stacked into an array of dimensions n by l, where n is the number of input vectors and l is the length of each vector. l must be same length as the last output layer of the net.
    - mini_batch_size (int): Length of the mini batches in stochastic gradient descent.
    - learning_rate (float): Learning rate in stochastic gradient descent. Usually small (< 1).
    - iterations (int): How many times to repeat stochastic gradient descent process. Reshuffles each iteration.
    """

    full_batch_size = len(X)
    shuffler = np.random.permutation(full_batch_size)
    X_shuffled = X[shuffler]
    Y_shuffled = Y[shuffler]
    for i in range(0, full_batch_size, mini_batch_size):
      slice_upper_bound = min(i + mini_batch_size, full_batch_size)
      X_mini_batch = X_shuffled[i : slice_upper_bound]
      Y_mini_batch = Y_shuffled[i : slice_upper_bound]
      self.run_batch(X_mini_batch)
      average_gradients = self.get_cost_gradient_batch_average(Y_mini_batch)
      average_cost_gradient_on_weight_layers, average_cost_gradient_on_bias1_layers, average_cost_gradient_on_bias2_layers = average_gradients
      for layer_i in range(-1, -self.num_layers, -1):
        self.weight_layers[layer_i] -= average_cost_gradient_on_weight_layers[layer_i] * learning_rate
        self.bias1_layers[layer_i] -= average_cost_gradient_on_bias1_layers[layer_i] * learning_rate
        if self.layer_is_xor[layer_i]:  self.bias2_layers[layer_i] -= average_cost_gradient_on_bias2_layers[layer_i] * learning_rate


