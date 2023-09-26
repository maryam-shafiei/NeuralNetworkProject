import time
import numpy as np


class NeuralNetwork:
    def __init__(self, sizes=[784, 16, 16, 10], epoch=20, lr=1, batch_size=10):
        self.sizes = sizes
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size

        self.params = {
            'w1': np.random.normal(size=(sizes[1], sizes[0])),    # 16x784
            'w2': np.random.normal(size=(sizes[2], sizes[1])),   # 16x16
            'w3': np.random.normal(size=(sizes[3], sizes[2])),   # 10x16
            'b1': np.zeros((sizes[1], 1)),
            'b2': np.zeros((sizes[2], 1)),
            'b3': np.zeros((sizes[3], 1))
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivation(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, input):
        a = [None] * 4
        z = [None] * 4
        params = self.params

        a[0] = input  # 784x1

        # input layer to hidden layer 1
        z[1] = (params['w1'] @ a[0]) + params['b1']  # 16x1
        a[1] = self.sigmoid(z[1])

        # hidden layer 1 to hidden layer 2
        z[2] = (params['w2'] @ a[1]) + params['b2']  # 16x1
        a[2] = self.sigmoid(z[2])

        # hidden layer 2 to output layer
        z[3] = (params['w3'] @ a[2]) + params['b3']  # 10x1
        a[3] = self.sigmoid(z[3])

        return a, z

    def back_propagation(self, train_set):
        epochs_cost = []
        start_time = time.time()
        for i in range(self.epoch):
            np.random.shuffle(train_set)
            batches = []
            for l in range(0, len(train_set), self.batch_size):
                batches.append(train_set[l: l + self.batch_size])
            for batch in batches:
                grads = {
                    'w1': np.zeros((self.sizes[1], self.sizes[0])),  # 16x784
                    'w2': np.zeros((self.sizes[2], self.sizes[1])),  # 16x16
                    'w3': np.zeros((self.sizes[3], self.sizes[2])),  # 10x16
                    'b1': np.zeros((self.sizes[1], 1)),
                    'b2': np.zeros((self.sizes[2], 1)),
                    'b3': np.zeros((self.sizes[3], 1))
                }
                for image, expected_output in batch:
                    a, z = self.feed_forward(image)

                    activation_grads = {
                        'a2': np.zeros((self.sizes[2], 1)),
                        'a1': np.zeros((self.sizes[1], 1))
                    }

                    # output layer
                    for j in range(grads['w3'].shape[0]):
                        for k in range(grads['w3'].shape[1]):
                            grads['w3'][j, k] += 2 * (a[3][j, 0] - expected_output[j, 0]) * self.sigmoid_derivation(z[3][j, 0]) * a[2][k, 0]

                    for j in range(grads['b3'].shape[0]):
                        grads['b3'][j, 0] += 2 * (a[3][j, 0] - expected_output[j, 0]) * self.sigmoid_derivation(z[3][j, 0])

                    # hidden layer 2
                    for k in range(16):
                        for j in range(10):
                            activation_grads['a2'][k, 0] += 2 * (a[3][j, 0] - expected_output[j, 0]) * self.sigmoid_derivation(z[3][j, 0]) * self.params['w3'][j, k]

                    for k in range(grads['w2'].shape[0]):
                        for m in range(grads['w2'].shape[1]):
                            grads['w2'][k, m] += activation_grads['a2'][k, 0] * self.sigmoid_derivation(z[2][k, 0]) * a[1][m, 0]

                    for k in range(grads['b2'].shape[0]):
                        grads['b2'][k, 0] += activation_grads['a2'][k, 0] * self.sigmoid_derivation(z[2][k, 0])

                    # hidden layer 1
                    for m in range(16):
                        for k in range(16):
                            activation_grads['a1'][m, 0] += activation_grads['a2'][k, 0] * self.sigmoid_derivation(z[2][k, 0]) * self.params['w2'][k, m]

                    for m in range(grads['w1'].shape[0]):
                        for v in range(grads['w1'].shape[1]):
                            grads['w1'][m, v] += activation_grads['a1'][m, 0] * self.sigmoid_derivation(z[1][m, 0]) * image[v, 0]

                    for m in range(grads['b1'].shape[0]):
                        grads['b1'][m, 0] += activation_grads['a1'][m, 0] * self.sigmoid_derivation(z[1][m, 0])

                self.update_weights(grads)
            epoch_cost = 0
            for inp in train_set:
                a, _ = self.feed_forward(inp[0])
                for j in range(0, self.sizes[3]):
                    epoch_cost += np.power((a[3][j, 0] - inp[1][j, 0]), 2)
            epoch_cost /= len(train_set)
            epochs_cost.append(epoch_cost)
        end_time = time.time()
        return self.compute_accuracy(train_set), end_time - start_time, epochs_cost

    def vectorization(self, train_set):
        epochs_cost = []
        start_time = time.time()
        for i in range(self.epoch):
            np.random.shuffle(train_set)
            batches = []
            for l in range(0, len(train_set), self.batch_size):
                batches.append(train_set[l: l + self.batch_size])
            for batch in batches:
                grads = {
                    'w1': np.zeros((self.sizes[1], self.sizes[0])),  # 16x784
                    'w2': np.zeros((self.sizes[2], self.sizes[1])),  # 16x16
                    'w3': np.zeros((self.sizes[3], self.sizes[2])),  # 10x16
                    'b1': np.zeros((self.sizes[1], 1)),
                    'b2': np.zeros((self.sizes[2], 1)),
                    'b3': np.zeros((self.sizes[3], 1))
                }
                for image, expected_output in batch:
                    a, z = self.feed_forward(image)

                    activation_grads = {
                        'a2': np.zeros((self.sizes[2], 1)),
                        'a1': np.zeros((self.sizes[1], 1))
                    }

                    # output layer
                    grads['w3'] += 2 * (a[3] - expected_output) * self.sigmoid_derivation(z[3]) @ np.transpose(a[2])
                    grads['b3'] += 2 * (a[3] - expected_output) * self.sigmoid_derivation(z[3])

                    # hidden layer 2
                    activation_grads['a2'] += np.transpose(self.params['w3']) @ (2 * (a[3] - expected_output) * self.sigmoid_derivation(z[3]))
                    grads['w2'] += (activation_grads['a2'] * self.sigmoid_derivation(z[2])) @ np.transpose(a[1])
                    grads['b2'] += activation_grads['a2'] * self.sigmoid_derivation(z[2])

                    # hidden layer 1
                    activation_grads['a1'] += np.transpose(self.params['w2']) @ (activation_grads['a2'] * self.sigmoid_derivation(z[2]))
                    grads['w1'] += (activation_grads['a1'] * self.sigmoid_derivation(z[1])) @ np.transpose(image)
                    grads['b1'] += activation_grads['a1'] * self.sigmoid_derivation(z[1])

                self.update_weights(grads)
            epoch_cost = 0
            for inp in train_set:
                a, _ = self.feed_forward(inp[0])
                for j in range(0, self.sizes[3]):
                    epoch_cost += np.power((a[3][j, 0] - inp[1][j, 0]), 2)
            epoch_cost /= len(train_set)
            epochs_cost.append(epoch_cost)
        end_time = time.time()
        return self.compute_accuracy(train_set), end_time - start_time, epochs_cost

    def update_weights(self, grads):
        for key, value in grads.items():
            self.params[key] -= (self.lr * (value / self.batch_size))

    def compute_accuracy(self, train_set):
        number_of_data = 0
        true_predictions = 0
        for data in train_set:
            number_of_data += 1
            output = self.feed_forward(data[0])[0][3]
            if np.argmax(output) == np.argmax(data[1]):
                true_predictions += 1
        return (true_predictions / number_of_data) * 100
