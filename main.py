import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from NeuralNetwork import NeuralNetwork

TRAIN_IMAGE_PATH = 'train-images.idx3-ubyte'
TRAIN_LABEL_PATH = 'train-labels.idx1-ubyte'
TEST_IMAGE_PATH = 't10k-images.idx3-ubyte'
TEST_LABEL_PATH = 't10k-labels.idx1-ubyte'


def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def load_data(images_path, labels_path):
    images_file = open(images_path, 'rb')
    images_file.seek(4)
    num_of_images = int.from_bytes(images_file.read(4), 'big')
    images_file.seek(16)

    labels_file = open(labels_path, 'rb')
    labels_file.seek(8)

    data_set = []
    for n in range(num_of_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(images_file.read(1), 'big') / 256

        label_value = int.from_bytes(labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        data_set.append((image, label))
    return data_set


train_set = load_data(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH)
test_set = load_data(TEST_IMAGE_PATH, TEST_LABEL_PATH)

# show_image(train_set[2][0])
# print(train_set[2][1])
# plt.show()

# second step
NN_2 = NeuralNetwork(sizes=[784, 16, 16, 10], epoch=20, lr=1, batch_size=10)
acc_forward = NN_2.compute_accuracy(train_set[:100])
print(acc_forward)

# third step
NN_3 = NeuralNetwork(sizes=[784, 16, 16, 10], epoch=20, lr=1, batch_size=10)
acc_backward, time_backward, epochs_cost = NN_3.backward_pass(train_set[:100])
print(f'acc: {acc_backward}, time: {time_backward}')
plt.plot(range(0, len(epochs_cost) + 1), epochs_cost)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()

# fourth step
NN_4 = NeuralNetwork(sizes=[784, 16, 16, 10], epoch=200, lr=1, batch_size=10)
acc_vector, time_vector, epochs_cost_vector = NN_4.vectorization(train_set[:100])
print(f'acc: {acc_vector}, time: {time_vector}')
plt.plot(range(0, len(epochs_cost_vector) + 1), epochs_cost_vector)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()

# fifth step
NN_5 = NeuralNetwork(sizes=[784, 16, 16, 10], epoch=5, lr=1, batch_size=50)
acc_vector, time_vector_test, epochs_cost_vector_test = NN_5.vectorization(train_set)
acc_test = NN_5.compute_accuracy(test_set)
print(f'acc: {acc_test}, time: {time_vector_test}')
plt.plot(range(1, len(epochs_cost_vector_test) + 1), epochs_cost_vector_test)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Test Set")
plt.show()