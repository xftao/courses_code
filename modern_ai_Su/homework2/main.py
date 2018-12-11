import csv
import numpy as np
import matplotlib.pyplot as plt


num_cluster = 4
print("haha")

def read_train_data():
    train_data = []
    with open('TrainingData_GMM.csv', 'r') as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            line = [float(num) for num in line]
            train_data.append(line)
    return train_data


def cal_prob(x, miu, sigma, dimension=2):
    exp_val = np.dot(np.subtract(x, miu).transpose(), np.linalg.inv(sigma))
    exp_val = np.dot(exp_val, np.subtract(x, miu))
    p = ((2 * np.pi) ** (dimension / 2) * np.linalg.det(sigma) ** (1 / 2)) ** (-1)
    p = p * np.exp((-1 / 2) * exp_val)
    return p
#
#
# def e_step(data, alpha, miu, sigma):
#     expect = np.zeros((4,))
#     for data_i in data:
#         for a_i, miu_i, sigma_i in zip(alpha, miu, sigma):


def train():
    data = read_train_data()
    data = np.array(data)
    # cal_prob([[1.5], [0]], [[0], [0]], [[1, 0], [0, 1]], 2)
    alpha = np.zeros((4, ))
    miu = np.zeros((4, 2))
    sigma = np.zeros((4, 2, 2))
    plt.plot(data[:, 0], data[:, 1], '.')
    plt.show()


if __name__ == '__main__':
    train()
