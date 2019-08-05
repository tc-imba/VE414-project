
1. density.jl
	read data_proj_414.csv
	write samples of Tayes into samples.csv
2. gmm.jl (implement GMM with EM algo.)
	read samples.csv
	fit gmm for k=10 to 30
	write position of the trees into trees.csv
3. run poisson.jl (require "Plots")
	read trees.csv
	plot trees in forest