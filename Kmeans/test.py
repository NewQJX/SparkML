from KmeansOnSpark import *
from sklearn.neighbors import KNeighborsClassifier

model = Kmeans()
data, target = model.load_iris()
centers = model.kmeans(data, num_iteration=3)
print(centers)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(centers, [0,2,1])
pred = knn.predict(data[:, :-1])
print("predict:" + str(pred))
print("target:" + str(target))
accuracy = np.sum(pred == target) / 150
print("accuracy is :" + str(accuracy))