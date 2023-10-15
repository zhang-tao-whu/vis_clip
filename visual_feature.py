from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# load feature
indexes = []
queries = np.load('./representations/refiner.npy')[0]

label = np.arange(0, queries.shape[2])
label = np.expand_dims(label, axis=0)
label = np.repeat(label, axis=0, repeats=queries.shape[1])

queries = np.transpose(queries, axes=(2, 1, 0))  # (q, t, c)
label = np.transpose(label, axes=(1, 0))  # (q, t)
nq, nt, nc = queries.shape
label = label.reshape((nq * nt, ))
queries = queries.reshape((nq * nt, nc, ))

pca = PCA()
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
plt.figure(figsize=(8, 6))
Xt = pipe.fit_transform(queries)
plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=label)
plt.show()