# load a couple of packages
using Distributions
using LinearAlgebra
using CSV
#using Pkg
#Pkg.add("Plots")
using Plots

num_observed_trees = 0
trees_x = Array{Float64,1}()
trees_y = Array{Float64,1}()
grid_observed = zeros(Int8, 107, 107)
csvFile = CSV.File("./data_proj_414.csv")
treeFile = CSV.File("./trees.csv")
num_observed_grid = 0
for (i, row) in enumerate(csvFile)
    global num_observed_grid
    x, y = ceil(Int64, row.X), ceil(Int64, row.Y)
    if grid_observed[x, y] == 0
        grid_observed[x, y] = 1
        num_observed_grid += 1
    end
end

for (i, row) in enumerate(treeFile)
    global num_observed_trees
    num_observed_trees += 1
    push!(trees_x, row.X)
    push!(trees_y, row.Y)
end

lambda_hat = num_observed_trees / num_observed_grid
grid_distribution = Poisson(lambda_hat)

sum_repeat = 0
for iter = 1:100
    global sum_repeat
    num_new_trees = 0
    for x = 1:107
        for y = 1:107
            if grid_observed[x, y] == 0
                r = rand()
                num_in_grid = 0
                while cdf(grid_distribution, num_in_grid) < r
                    num_in_grid = num_in_grid + 1
                end
                if iter == 1
                    for k = 1:num_in_grid
                        push!(trees_x, x-rand())
                        push!(trees_y, y-rand())
                    end
                end
                num_new_trees += num_in_grid
            end
        end
    end
    sum_repeat += num_new_trees
end
println("Trees in unobserved area: ", round(Int, sum_repeat/100))
println("Total trees number in forest: ", round(Int, sum_repeat/100)+num_observed_trees)
plot(trees_x, trees_y, seriestype=:scatter, title="Trees in the Forest")
