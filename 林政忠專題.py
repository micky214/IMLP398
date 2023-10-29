import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X_train, X_test = [], []
Y_train, Y_test = [], []
for i in range(1, 41):
    for j in range(1, 10):
        img = cv2.imread("data/{}_{}.png".format(i, j), cv2.IMREAD_GRAYSCALE)
        X_train.append(img)
        Y_train.append(i)
    img = cv2.imread("data/{}_10.png".format(i), cv2.IMREAD_GRAYSCALE)
    X_test.append(img)
    Y_test.append(i)
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

x = X_train.reshape(len(Y_train), -1)
print(x.shape)

pca = PCA()
pca.fit(X_train.reshape(len(Y_train), -1))

plt.figure(figsize=(16, 20))
plt.subplot(1, 5, 1)
plt.axis("off")
plt.title("Mean face")
plt.imshow(pca.mean_.reshape(X_train[0].shape), cmap="gray")

for i in range(4):
    plt.subplot(1, 5, i + 2)
    plt.axis("off")
    plt.title("Eigenface{}".format(i + 1))
    plt.imshow(pca.components_[i].reshape(X_train[0].shape), cmap="gray")
    plt.show()
    img = cv2.imread("data/2_1.png", cv2.IMREAD_GRAYSCALE).reshape(1, -1)
a = pca.transform(img)

plt.figure(figsize=(16, 12))
n_components = [3, 50, 170, 240, 345]
for i, n in enumerate(n_components):
    face = np.zeros(img.shape)
    for j in range(n):
        face = face + a[0][j] * pca.components_[j]
    face = face + pca.mean_
    MSE = np.mean((face - img) ** 2)
    plt.subplot(1, 5, i + 1)
    plt.axis("off")
    plt.title("n={}, MSE ={:.2f}".format(n, MSE))
    plt.imshow(face.reshape(X_train[0].shape), cmap="gray")

reduced_X_train = pca.transform(X_train.reshape(len(Y_train), -1))
K = [1, 3, 5]
N = [3, 50, 170]
random = np.random.permutation(len(Y_train))
reduced_X_train = reduced_X_train[random]
Y_train_random = Y_train[random]
for k in K:
    print("k={}".format(k))
    knn = KNeighborsClassifier(n_neighbors=k)
    for n in N:
        print("  n={}, ".format(n), end="")
        score = cross_val_score(knn, reduced_X_train[:, :n], Y_train_random, cv=3)
        print("score{:.4f}".format(score.mean()))
        k = 1 
n = 50
reduced_X_test = pca.transform(X_test.reshape(len(Y_test),-1))

knn = KNeighborsClassifier(n_neighbors = k )
knn.fit(reduced_X_train[:, :n] , Y_train_random)

print ('accuracy = {}'.format (knn . score(reduced_X_test[: , :n ], Y_test)))
