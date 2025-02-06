

import numpy as np
import matplotlib.pyplot as plt
import time
from xoral_net import XoralNet

plt.style.use('dark_background')



def test_xor_dataset():

  def test_net_training(
      net: XoralNet,
      X_train: np.ndarray,
      Y_train: np.ndarray,
      X_test: np.ndarray,
      Y_test: np.ndarray,
      until: float = 0.9,
      mode: str = 'a'):

    num_iters = 1
    iters_elapsed = 0
    net.run_batch(X_test)
    metric = round(get_metric(net, Y_test, mode), 2)
    print(f"{iters_elapsed} iters: {'Accuracy' if mode == 'a' else "Cost"}: {metric}")
    t0 = time.time()
    while (mode == 'a' and metric < until) or (mode == 'c' and metric > until):
      net.train(X_train, Y_train, 10, iterations = num_iters, learning_rate = 0.01)
      net.run_batch(X_test)
      new_metric = round(get_metric(net, Y_test, mode), 2)
      iters_elapsed += num_iters
      if new_metric != metric:
        metric = new_metric
        print(f"{iters_elapsed} iters: {'Accuracy' if mode == 'a' else "Cost"}: {metric}")
    print(f"Time: {time.time() - t0}")
    print()

  def get_metric(
      net: XoralNet,
      Y: np.ndarray,
      mode: str = 'a'):

    metric = net.get_accuracy(net.vectors_to_classes(Y)) if mode == 'a' else net.get_cost_batch_average(Y)
    return metric

  n = 1000
  X = np.random.randn(n, 2) / 5 + (np.random.randint(2, size = (n, 2)) * 2 - 1)
  Y = np.array([[0] if X[i][0] * X[i][1] > 0 else [1] for i in range(len(X))])
  training_index_cutoff = int(0.9 * n)
  X_train = X[0 : training_index_cutoff]
  Y_train = Y[0 : training_index_cutoff]
  X_test = X[training_index_cutoff :]
  Y_test = Y[training_index_cutoff :]

  print()

  nnet = XoralNet([2, 2, 1])
  print("Training Neural Net on XOR dataset:")
  test_net_training(nnet, X_train, Y_train, X_test, Y_test)

  print()

  xnet = XoralNet([2, 1], xor_layers = [1])
  print("Training Xoral Net on XOR dataset:")
  test_net_training(xnet, X_train, Y_train, X_test, Y_test)

  print()



def main():

  test_xor_dataset()

if __name__ == '__main__': main()
