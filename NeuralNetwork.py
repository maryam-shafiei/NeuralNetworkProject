import time
import numpy as np


class NeuralNetwork:
    def __init__(self, sizes=[784, 16, 16, 10], epoch=20, lr=1, batch_size=10):
        self.sizes = sizes
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size

        input_layer = sizes[0]
        hidden_1 = sizes[1]
        hidden_2 = sizes[2]
        output_layer = sizes[3]

        self.params = {
            'w1': np.random.normal(size=(hidden_1, input_layer)),    # 16x784
            'w2': np.random.normal(size=(hidden_2, hidden_1)),   # 16x16
            'w3': np.random.normal(size=(output_layer, hidden_2)),   # 10x16
            'b1': np.zeros((hidden_1, 1)),
            'b2': np.zeros((hidden_2, 1)),
            'b3': np.zeros((output_layer, 1))
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_pass(self, input):
        a = [None] * 4
        z = [None] * 4
        params = self.params

        a[0] = input # 784x1

        # input layer to hidden_1
        z[1] = (params['w1'] @ a[0]) + params['b1'] # 16x1
        a[1] = self.sigmoid(z[1])

        # hidden_1 to hidden_2
        z[2] = (params['w2'] @ a[1]) + params['b2'] # 16x1
        a[2] = self.sigmoid(z[2])

        # hidden_2 to output layer
        z[3] = (params['w3'] @ a[2]) + params['b3'] # 10x1
        a[3] = self.sigmoid(z[3])

        return a, z

    def backward_pass(self, train_set):
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
                    # compute the output (image is equal to a0)
                    a, z = self.forward_pass(image)

                    # ---- Last layer
                    # weight
                    for j in range(grads['w3'].shape[0]):
                        for k in range(grads['w3'].shape[1]):
                            grads['w3'][j, k] += 2 * (a[3][j, 0] - expected_output[j, 0]) * self.sigmoid_der(z[3][j, 0]) * a[2][k, 0]

                    # bias
                    for j in range(grads['b3'].shape[0]):
                        grads['b3'][j, 0] += 2 * (a[3][j, 0] - expected_output[j, 0]) * self.sigmoid_der(z[3][j, 0])

                    # ---- 3rd layer
                    # activation
                    delta_3 = np.zeros((16, 1))
                    for k in range(16):
                        for j in range(10):
                            delta_3[k, 0] += 2 * (a[3][j, 0] - expected_output[j, 0]) * self.sigmoid_der(z[3][j, 0]) * self.params['w3'][j, k]

                    # weight
                    for k in range(grads['w2'].shape[0]):
                        for m in range(grads['w2'].shape[1]):
                            grads['w2'][k, m] += delta_3[k, 0] * self.sigmoid_der(z[2][k, 0]) * a[1][m, 0]

                    # bias
                    for k in range(grads['b2'].shape[0]):
                        grads['b2'][k, 0] += delta_3[k, 0] * self.sigmoid_der(z[2][k, 0])

                    # ---- 2nd layer
                    # activation
                    delta_2 = np.zeros((16, 1))
                    for m in range(16):
                        for k in range(16):
                            delta_2[m, 0] += delta_3[k, 0] * self.sigmoid_der(z[2][k, 0]) * self.params['w2'][k, m]

                    # weight
                    for m in range(grads['w1'].shape[0]):
                        for v in range(grads['w1'].shape[1]):
                            grads['w1'][m, v] += delta_2[m, 0] * self.sigmoid_der(z[1][m, 0]) * image[v, 0]

                    # bias
                    for m in range(grads['b1'].shape[0]):
                        grads['b1'][m, 0] += delta_2[m, 0] * self.sigmoid_der(z[1][m, 0])

                self.update_weights(grads)
            epoch_cost = 0
            for inp in train_set:
                a, _ = self.forward_pass(inp[0])
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
                    # compute the output (image is equal to a0)
                    a, z = self.forward_pass(image)

                    # ---- Last layer
                    # weight
                    grads['w3'] += 2 * (a[3] - expected_output) * self.sigmoid_der(z[3]) @ np.transpose(a[2])

                    # bias
                    grads['b3'] += 2 * (a[3] - expected_output) * self.sigmoid_der(z[3])

                    # ---- 3rd layer
                    # activation
                    delta_3 = np.zeros((16, 1))
                    delta_3 += np.transpose(self.params['w3']) @ (2 * (a[3] - expected_output) * self.sigmoid_der(z[3]))

                    # weight
                    grads['w2'] += (delta_3 * self.sigmoid_der(z[2])) @ np.transpose(a[1])

                    # bias
                    grads['b2'] += delta_3 * self.sigmoid_der(z[2])

                    # ---- 2nd layer
                    # activation
                    delta_2 = np.zeros((16, 1))
                    delta_2 += np.transpose(self.params['w2']) @ (delta_3 * self.sigmoid_der(z[2]))

                    # weight
                    grads['w1'] += (delta_2 * self.sigmoid_der(z[1])) @ np.transpose(image)

                    # bias
                    grads['b1'] += delta_2 * self.sigmoid_der(z[1])

                self.update_weights(grads)
            epoch_cost = 0
            for inp in train_set:
                a, _ = self.forward_pass(inp[0])
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
            output = self.forward_pass(data[0])[0][3]
            if np.argmax(output) == np.argmax(data[1]):
                true_predictions += 1
        return (true_predictions / number_of_data) * 100