using Distributions
using Random
using LinearAlgebra
using CSV

function get_pdf(sample, mu, sigma)
    try
        distribution = MvNormal(mu, sigma)
        return pdf(distribution, sample)
    catch
        # println(sigma)
        return 0
    end
end

function get_log_likelihood(data, k, mu, sigma, gamma)
    res = 0.0
    for i=1:size(data)[1]
        cur = 0.0
        for j=1:length(k)
            cur += gamma[j,i] * get_pdf(data[i], mu[j], sigma[j])
        end
        res += log(cur)
    end
    return res
end

function em(data, k, mu, sigma, steps=1000)
    num_gau = length(k)  # 高斯分布个数
    num_data = size(data)[1]  # 数据个数
    gamma = zeros(num_gau, num_data)  # gamma[j][i]表示第i个样本点来自第j个高斯模型的概率
    likelihood_record = []  # 记录每一次迭代的log-likelihood值
    temp = zeros(num_gau)
    # println(get_log_likelihood(data, k, mu, sigma, gamma))

    for step=1:steps
        # 计算gamma矩阵
        for j=1:num_data
            for i=1:num_gau
                # println(sigma[i])
                temp[i] = get_pdf(data[j], mu[i], sigma[i]) * k[i]
            end
            temp_sum = sum(temp)
            if temp_sum == 0
                for i=1:num_gau
                    gamma[i,j] = 1.0 / num_gau
                end
            else
                for i=1:num_gau
                    gamma[i,j] = temp[i] / temp_sum
                end
            end
                # gamma[i][j] = k[i] * get_pdf(data[j], mu[i], sigma[i]) /
                             # sum([k[t] * get_pdf(data[j], mu[t], sigma[t]) for t in range(num_gau)])
        end
        cur_likelihood = get_log_likelihood(data, k, mu, sigma, gamma)  # 计算当前log-likelihood
        push!(likelihood_record, cur_likelihood)
        # 更新mu
        for i=1:num_gau
            # println(gamma[i,:])
            # println(data)
            # println(sum(gamma[i,:]))
            for j=1:2
                mu[i][j] = 0
                for n=1:num_data
                    mu[i][j] += gamma[i,n] * data[n][j]
                end
                mu[i][j] /= sum(gamma[i,:])
            end
            # mu[i] = (gamma[i,:]'*data) ./ sum(gamma[i,:])
        end
        # 更新sigma
        for i=1:num_gau
            # cov = [np.dot((data[t] - mu[i]).reshape(-1, 1), (data[t] - mu[i]).reshape(1, -1)) for t in range(num_data)]
            cov_sum = zeros(2, 2)
            for j=1:num_data
                A = data[j] - mu[i]
                cov_sum += gamma[i,j] .* (A * A')
            end
            # println(cov_sum)
            if (cov_sum[1,1] > 0 && cov_sum[1,2] > 0 && cov_sum[2,1] > 0 && cov_sum[2,2] > 0)
                sigma[i] = cov_sum ./ sum(gamma[i,:])
            else
                # sigma[i] = [1. 0.]; [0. 1.]
            end
        end
        # 更新k
        for i=1:num_gau
            k[i] = sum(gamma[i,:]) / num_data
        end
        # println(step, " ", cur_likelihood)
        # print('step: {}\tlikelihood:{}'.format(step + 1, cur_likelihood))
    end
    return k, mu, sigma, gamma, likelihood_record
end

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

function gmm(data,k)
    k_init, mu_init, sigma_init = init(k)
    k_res, mu_res, sigma_res, gamma, likelihood_record = em(data, k_init, mu_init, sigma_init, 30)
    return likelihood_record[30]
end

data = []
csvFile = CSV.File("samples.csv", header=false)
for (i, row) in enumerate(csvFile)
    push!(data, [row.Column1, row.Column2])
end

for k=1:30
    likelihood = gmm(data, k)
    println(k, " ", likelihood)
end

# k_init, mu_init, sigma_init = init(10)
# k_res, mu_res, sigma_res, gamma, likelihood_record = em(data, k_init, mu_init, sigma_init, 50)
