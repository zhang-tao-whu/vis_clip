from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# load feature
#indexes = [61, 0, 65, 32, 42, 90, 70, 88, 58, 81, 4, 3, 54, 45, 18]
# indexes = [61, 0, 58, 65, 42, 32, 81, 90, 70, 88, 45, 4, 3, 18]
indexes = [32, 61, 9]
name = 'tracker'
suffix = '_cl'
#suffix = ''

queries = np.load('./representations/{}.npy'.format(name))[0][:, :, indexes]

label = np.arange(0, queries.shape[2])
label = np.expand_dims(label, axis=0)
label = np.repeat(label, axis=0, repeats=queries.shape[1])

queries = np.transpose(queries, axes=(2, 1, 0))  # (q, t, c)
label = np.transpose(label, axes=(1, 0))  # (q, t)
nq, nt, nc = queries.shape
label = label.reshape((nq * nt, ))
queries = queries.reshape((nq * nt, nc, ))

pca = PCA(n_components=2)
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
plt.figure(figsize=(8, 6))
Xt = pipe.fit_transform(queries)

print(Xt.shape)
plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=label, cmap='Spectral', s=8)
plt.xticks([])    # 去 x 轴刻度
plt.yticks([])
plt.savefig('./representations/{}.PDF'.format(name + suffix), bbox_inches='tight', pad_inches=-0.1)
plt.show()