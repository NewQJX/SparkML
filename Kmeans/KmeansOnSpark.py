import numpy as np
from sklearn.datasets import load_iris
from pyspark.sql import SparkSession

class Kmeans:
	def __init__(self):
		pass
		
	def _getNearest(self, a, centers):
		if len(a) != centers.shape[1]:
			print("array size mismatch!!!")
			return -1
		result = -1
		dist = 10000
		for i in np.arange(len(centers)):
			temp = np.linalg.norm(a - centers[i])
			if temp < dist:
				dist = temp
				result = i
		return result

	def _init_sparkContext(self):
		spark = SparkSession.builder.appName("Kmeans").master("local").getOrCreate()
		sc = spark.sparkContext
		return sc
		
	def load_iris(self):
		iris = load_iris()
		data = iris.data
		target = iris.target
		data = np.c_[data, np.ones((data.shape[0],1))]
		return data, target
		
	def kmeans(self, data, num_iteration = 10, print_log = True):
		sc = self._init_sparkContext()							#initialize sparkContext
		centers = data[[33, 93, 123]][:, :-1]					#initialize cluster centers			
		dataRDD = sc.parallelize(data)
		
		for i in range(num_iteration):
			nearestRDD = dataRDD.map(lambda a : (self._getNearest(a[:-1], centers), a))
			sumRDD = nearestRDD.reduceByKey(lambda a, b : a + b)
			centersRDD = sumRDD.map(lambda a : a[1][:-1] / a[1][-1])
			centers = np.array(centersRDD.collect())
			if i % 2 == 0 and print_log == True:
				print("centers after iteration of " + str(i) + " is :"+str(centers))
		return centers

