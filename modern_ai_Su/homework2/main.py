import csv
import numpy as np
import matplotlib.pyplot as plt


num_cluster = 4
num_data = 5000
data_dimension = 2


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


def m_step(alpha, data, gamma, miu, sigma, ):
    n_k = np.zeros((num_cluster,))
    for k in range(num_cluster):
        # calculate nk
        for gamma_k_i in gamma[:, k]:
            n_k[k] = n_k[k] + gamma_k_i

        # update miu
        temp_miu = np.zeros(num_cluster, data_dimension)
        for i in range(num_data):
            temp_miu = temp_miu + gamma[k] * data[i]

        miu[k] = temp_miu / n_k[k]


def e_step(data, alpha, miu, sigma):
    gamma = np.zeros((num_cluster, num_data))
    for i, data_i in enumerate(data):
        for k, a_k, miu_k, sigma_k in enumerate(zip(alpha, miu, sigma)):
            gamma[i, k] = a_k * cal_prob(data_i, miu_k, sigma_k)

        k_sum = np.sum(gamma[i, :])
        for k, prob_k in enumerate(gamma[i, :]):
            gamma[i, k] = prob_k/k_sum

    return gamma


def train():
    data = read_train_data()
    data = np.array(data)
    # cal_prob([[1.5], [0]], [[0], [0]], [[1, 0], [0, 1]], 2)
    alpha = np.array([1/num_cluster, 1/num_cluster, 1/num_cluster, 1/num_cluster])
    miu = np.zeros((num_cluster, data_dimension))
    sigma = np.zeros((num_cluster, data_dimension, data_dimension))
    plt.plot(data[:, 0], data[:, 1], '.')
    plt.show()


if __name__ == '__main__':
    train()
