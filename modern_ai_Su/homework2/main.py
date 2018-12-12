import csv
import numpy as np
import matplotlib.pyplot as plt


num_cluster = 4
num_data = 5000
data_dimension = 2
initial_sigma = [[0.5, 0], [0, 0.5]]


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


def m_step(alpha, data, gamma, miu, sigma):
    n_k = np.zeros((num_cluster,))
    for k in range(num_cluster):
        # calculate nk
        for gamma_k_i in gamma[:, k]:
            n_k[k] = n_k[k] + gamma_k_i

        # update miu
        temp_miu = np.zeros((1, data_dimension))
        for i in range(num_data):
            temp_miu = temp_miu + gamma[i, k] * data[i, :]
        miu[k] = np.divide(temp_miu, n_k[k])

        # update sigma
        temp_sigma = np.zeros((data_dimension, data_dimension))
        for i in range(num_data):
            distance = np.subtract(data[i, :], miu[k])
            distance = np.expand_dims(distance, axis=0)
            single_data_sigma = np.matmul(distance.transpose(), distance)
            temp_sigma = np.add(temp_sigma, np.multiply(gamma[i, k], single_data_sigma))
        sigma[k] = np.divide(temp_sigma, n_k[k])

    # update alpha
    for k in range(num_cluster):
        alpha[k] = n_k[k] / num_data


def e_step(data, alpha, miu, sigma):
    gamma = np.zeros((num_data, num_cluster))
    for i, data_i in enumerate(data):
        for k in range(num_cluster):
            gamma[i, k] = alpha[k] * cal_prob(data_i, miu[k], sigma[k])

        k_sum = np.sum(gamma[i, :])
        for k, prob_k in enumerate(gamma[i, :]):
            gamma[i, k] = prob_k/k_sum

    return gamma


def train(data):
    # plt.plot(data[:, 0], data[:, 1], '.')
    # plt.show()
    # cal_prob([[1.5], [0]], [[0], [0]], [[1, 0], [0, 1]], 2)
    alpha = np.array([1/num_cluster, 1/num_cluster, 1/num_cluster, 1/num_cluster])

    # miu initial guess
    index = np.random.randint(0, 5001, 4)
    miu = np.zeros((num_cluster, data_dimension))
    for i in range(num_cluster):
        miu[i] = data[index[i]]

    # sigma initial guess
    sigma = []
    for i in range(num_cluster):
        sigma.append(initial_sigma)
    sigma = np.array(sigma)
    step = 0
    while 1:
        step = step + 1
        old_miu = miu.copy()
        gamma = e_step(data, alpha, miu, sigma)
        m_step(alpha, data, gamma, miu, sigma)
        l2_dist = 0
        for i in range(num_cluster):
            temp_val = np.linalg.norm(np.subtract(old_miu[i], miu[i]))
            if temp_val > l2_dist:
                l2_dist = temp_val

        print(step)
        if l2_dist < 1e-3 or step == 200:
            break

    return alpha, miu, sigma


def perform(data, alpha, miu, sigma):
    selected_point = [data[i, :] for i in np.random.randint(0, 5000, 500)]
    cluster = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    for point in selected_point:
        prob = []
        for k in range(num_cluster):
            prob.append(alpha[k] * cal_prob(point, miu[k], sigma[k]))

        cluster[np.asscalar(np.argmax(prob))].append(point.tolist())

    plt_arg = ['.r', '.b', '.g', '.y']
    for key in cluster:
        cluster[key] = np.array(cluster[key])
        plt.plot(cluster[key][:, 0], cluster[key][:, 1], plt_arg[key])
    plt.show()


def run():
    data = read_train_data()
    data = np.array(data)
    alpha, miu, sigma = train(data)
    perform(data, alpha, miu, sigma)



if __name__ == '__main__':
    run()
