import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import random


def get_pdf(sample, mu, sigma):
    res = stats.multivariate_normal(mu, sigma).pdf(sample)
    return res


def get_log_likelihood(data, k, mu, sigma, gamma):
    res = 0.0
    for i in range(len(data)):
        cur = 0.0
        for j in range(len(k)):
            cur += gamma[j][i] * get_pdf(data[i], mu[j], sigma[j])
        res += math.log(cur)
    return res


def em(data, k, mu, sigma, steps=1000):
    num_gau = len(k)  # 高斯分布个数
    num_data = data.shape[0]  # 数据个数
    gamma = np.zeros((num_gau, num_data))  # gamma[j][i]表示第i个样本点来自第j个高斯模型的概率
    likelihood_record = []  # 记录每一次迭代的log-likelihood值
    for step in range(steps):
        # 计算gamma矩阵
        for i in range(num_gau):
            for j in range(num_data):
                gamma[i][j] = k[i] * get_pdf(data[j], mu[i], sigma[i]) / \
                             sum([k[t] * get_pdf(data[j], mu[t], sigma[t]) for t in range(num_gau)])
        cur_likelihood = get_log_likelihood(data, k, mu, sigma, gamma)  # 计算当前log-likelihood
        likelihood_record.append(cur_likelihood)
        # 更新mu
        for i in range(num_gau):
            mu[i] = np.dot(gamma[i], data) / np.sum(gamma[i])
        # 更新sigma
        for i in range(num_gau):
            cov = [np.dot((data[t] - mu[i]).reshape(-1, 1), (data[t] - mu[i]).reshape(1, -1)) for t in range(num_data)]
            cov_sum = np.zeros((2, 2))
            for j in range(num_data):
                cov_sum += gamma[i][j] * cov[j]
            sigma[i] = cov_sum / np.sum(gamma[i])
        # 更新k
        for i in range(num_gau):
            k[i] = np.sum(gamma[i]) / num_data
        print('step: {}\tlikelihood:{}'.format(step + 1, cur_likelihood))
    return k, mu, sigma, gamma, likelihood_record

def init(k):
	k_init = [1/k]*k
	mu_init = []
	for i in range(k):
		mu_init.append([random.randint(0,107),random.randint(0,107)])
	sigma_init = [[[1, 0], [0, 1]] for i in range(k)]
	return k_init, mu_init, sigma_init

# data: [[x1,y1],...,[xn,yn]]
# k number of trees
def gmm(data,k):
    data = np.array(data)
    # 原始数据散点图
    # plt.scatter(data[:, 0], data[:, 1], s=5, c='r')
    # plt.title('Raw Data')
    # plt.show()
    # GMM参数初始值
    k_init, mu_init, sigma_init = init(k)
    k_res, mu_res, sigma_res, gamma, likelihood_record = em(data, k_init, mu_init, sigma_init, steps=30)  # EM算法
    # 根据EM算法的结果画分类图
    classify = gamma.argmax(axis=0)  # 计算每个样本点所属的单高斯模型
    k_gau = [[] for i in range(k)]  # 把数据集分成k份，每份属于一个单高斯模型
    for i in range(len(classify)):
        k_gau[classify[i]].append(data[i])
    # 转成numpy，方便画散点图
    k_gau = np.array(k_gau)
    for i in range(len(k_gau)):
        k_gau[i] = np.array(k_gau[i])
    colors = ['r', 'g', 'k', 'b']
    print(k_gau)
    for i in range(len(k_res)):
        plt.scatter(k_gau[i][:, 0], k_gau[i][:, 1], s=5, c=colors[i])
    plt.title('After EM')
    plt.show()
    # 画likelihood的变化曲线图
    plt.plot([n for n in range(len(likelihood_record))], likelihood_record)
    plt.title('step-likelihood')
    plt.xlabel('step')
    plt.ylabel('log-likelihood')
    plt.show()
    print('result:\nk: {}\nmu: {}\nsigma: {}'.format(k_res, mu_res, sigma_res))

