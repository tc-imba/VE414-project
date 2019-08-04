import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.mixture import GaussianMixture as GMM

df = pd.read_csv('data_proj_414.csv')
fden = np.zeros((108,108),dtype = np.float64)
rnum = np.zeros((108,108))


for index, row in df.iterrows():
	x = math.floor(row['X'])
	y = math.floor(row['Y'])
	if row['Close'] != 0:
		fden[x][y] += row['Close']/math.pi
		rnum[x][y] += 1
	if row['Far'] != 0:
		fden[x-1][y-1] += row['Far']/(8*math.pi)
		rnum[x-1][y-1] += 1
		fden[x-2][y-2] += row['Far']/(8*math.pi)
		rnum[x-2][y-2] += 1
	x = math.ceil(row['X'])
	y = math.ceil(row['Y'])
	if row['Close'] != 0:
		fden[x][y] += row['Close']/math.pi
		rnum[x][y] += 1
	if row['Far'] != 0:
		fden[x+1][y+1] += row['Far']/(8*math.pi)
		rnum[x+1][y+1] += 1
		fden[x+2][y+2] += row['Far']/(8*math.pi)
		rnum[x+2][y+2] += 1


ddf = {'X':[],'Y':[]}
for i in range(108):
	for j in range(108):
		if rnum[i][j]!=0:
			fden[i][j] /= rnum[i][j]
		for k in range(int(fden[i][j])*5):
			ddf['X'].append(i)
			ddf['Y'].append(j)


# sns.jointplot('X','Y',ddf,kind='kde')
# heatmap = sns.heatmap(fden,cmap='BuPu')

# ag = plt.scatter(df['X'],df['Y'],c=df['Close']+df['Far'],s=10,cmap='YlGnBu')
# plt.colorbar(ag)

dmatrix = pd.DataFrame(ddf).as_matrix(columns=None)
print(dmatrix)
gmm = GMM(n_components = 72).fit(dmatrix)
labels = gmm.predict(dmatrix)
plt.scatter(dmatrix[:,0],dmatrix[:,1],c=labels ,s=20,cmap='viridis')
plt.xlim(0,108)
plt.ylim(0,108)

# n_components = np.arange(70,80)
# models = [GMM(n, covariance_type='full', random_state=0).fit(dmatrix)
#           for n in n_components]
# plt.plot(n_components, [m.bic(dmatrix) for m in models], label='BIC')
# plt.plot(n_components, [m.aic(dmatrix) for m in models], label='AIC')
# plt.legend(loc='best')
plt.show()

