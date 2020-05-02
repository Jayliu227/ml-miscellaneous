import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

A = np.random.normal(scale=1, size=(100, 3))
B = np.array([x for x in np.random.normal(scale=5, size=(500, 3)) if np.linalg.norm(x) > 7])

fig = plt.figure()

ax = fig.add_subplot(2, 2, 1, projection='3d', aspect=1.0)

ax.scatter(A[:, 0], A[:, 1], A[:, 2])
ax.scatter(B[:, 0], B[:, 1], B[:, 2])
plt.title('Original Data Plot')

X = np.r_[A, B]  # row-wise merging, dimension of the row remains

# PCA decomposition
X2 = PCA(n_components=2).fit_transform(X)  # reduce to two dimension
A2 = X2[:A.shape[0], :]
B2 = X2[A.shape[0]:, :]

pca = plt.subplot(2, 2, 2, aspect=1.0)
pca.scatter(A2[:, 0], A2[:, 1])
pca.scatter(B2[:, 0], B2[:, 1])
pca.axis('equal')
plt.title('PCA')

# t-SNE
X3 = TSNE(n_components=2).fit_transform(X)
A3 = X3[:A.shape[0], :]
B3 = X3[A.shape[0]:, :]

t_sne = plt.subplot(2, 2, 4, aspect=1.0)
t_sne.scatter(A3[:, 0], A3[:, 1])
t_sne.scatter(B3[:, 0], B3[:, 1])
t_sne.axis('equal')
plt.title('t-SNE')

plt.show()
