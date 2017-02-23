from sklearn.cluster import KMeans
import numpy as np
import csv
import math
pos = []
lat0 = 0
label_id = []
test_num = 100000
node_num = 10
score = [[0]*24 for i in range(node_num)]

def transform(lat, lon):
	er = 6378137.0
	s = np.cos(lat0 * math.pi/180)
	tx = s * er * math.pi *lon / 180.0
	ty = s * er * np.log(np.tan((90.0 + lat) * math.pi / 360.0))
	# tx = lon * math.pi * er / 180.0
	# ty = er * np.log(np.tan((90.0 + lat) * math.pi / 360.0))
	l = [tx,ty]
	return l


def read_csv():
	file=open("yellow_tripdata_2016-01.csv", "r")
	reader = csv.reader(file)
	count = 0
	flag = 0
	for line in reader:
		if (line[1][9]!=str(1) or line[5]==str(0)):
			continue
		# TODO 
		if (count > test_num):
			break
		if (flag!=0):
			lat0 = float(line[5])
			flag = 1
		pos.append(transform(float(line[5]),float(line[6])))
		# pos.append(float[line5])
		# print pos[count]
		count +=1 
		
def Kmeans_cluster(X,n_clusters):
    X=np.array(X)
    kmeans=KMeans(n_clusters=n_clusters,random_state=0).fit(X)
    # print kmeans.cluster_centers_
    label_id = kmeans.labels_.tolist()
    # print label_id
    return label_id

def sum_up(label_id):
	file = open("yellow_tripdata_2016-01.csv", "r")
	reader = csv.reader(file)
	count = 0
	for line in reader:
		if (line[1][9]!=str(1) or line[5]==str(0)):
			continue
		if (count > test_num):
			break
		# 2016-01-01 00:00:00
		# 0123456789
		time = int(line[1][11:13])
		print time
		score[label_id[count]][time] += int(line[3])
		count += 1


def main():
	read_csv()
	X=[[1,2],[2,3],[1,1],[3,3]]
	label_id = Kmeans_cluster(pos,n_clusters=node_num)
	# print Kmeans_cluster(pos,n_clusters=node_num)
	sum_up(label_id)
	print score


if __name__=='__main__':
    main()
