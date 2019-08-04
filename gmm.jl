using Distributions
using Random

function get_pdf(sample, mu, sigma)
    distribution = MvNormal(mu, sigma)
    return pdf(distribution, sample)
end

function get_log_likelihood(data, k, mu, sigma, gamma)
    res = 0.0
    for i=1:length(data)
        cur = 0.0
        for j=1:length(k)
            cur += gamma[j][i] * get_pdf(data[i], mu[j], sigma[j])
        end
        res += log(cur)
    end
    return res
end

function em(data, k, mu, sigma, steps=1000)
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

function init(k)
	k_init = Array{Float64,1}()
	mu_init = []
    sigma_init = []
	for i=1:k
        push!(k_init, 1.0/k)
        push!(mu_init, rand(1.:107.,2))
        push!(sigma_init, [[1. 0.]; [0. 1.]])
    end
	return k_init, mu_init, sigma_init
end

k_init, mu_init, sigma_init = init(10)

get_pdf([1, 0], mu_init[1], sigma_init[1])


a = [[1. 0.]; [0. 1.]]

get_pdf([1, 0], [0. , 0.], a)
