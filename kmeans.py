from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import pylab as pl
#TODO
#python
#output minscore to the file

#c++
#read data
#vector<real_vetex> v to store the positions
#vector<vector<int> > score to store the score function (resolution: 1min)
#call function collect_data() get the real vertex
#get the score through extern variable score
#init graph with the score function and v


pos = []
lat0 = 0
label_id = []
test_num = 400000
node_num = 50
time_step = 1 # 10 mins
# score_min = [0] * (60/time_step)
# score_hour = [score_min for i in range(24)]
# score = [score_hour for i in range(node_num)]
score = np.zeros((node_num,24,60/time_step))
score_hour = [[0]*24 for i in range(node_num)]
lat = []
lng = []
mean_x = 0 # center y of map
mean_y = 0 # center x of map
center = [] # center coordinate of clusters
def transform(lat, lon):
	er = 6378137.0
	s = np.cos(lat0 * math.pi/180)
	tx = s * er * math.pi *lon / 180.0
	ty = s * er * np.log(np.tan((90.0 + lat) * math.pi / 360.0))
	l = [tx,ty]
	return l 	


def read_csv():
	file=open("yellow_tripdata_2016-01.csv", "r")
	reader = csv.reader(file)
	count = 0
	flag = 0
	for line in reader:
		count +=1 
		if (line[1][9]!=str(1) or line[5]==str(0)):
			continue
		# TODO 
		if (count > test_num):
			break
		if (flag!=0):
			lat0 = float(line[5])
			flag = 1
		pos.append(transform(float(line[6]),float(line[5])))


def update_pos(pos):
	sumx = 0
	sumy = 0
	new_pos = []
	for subpos in pos:
		sumx += subpos[0]
		sumy += subpos[1]

	global mean_x, mean_y
	mean_x = sumx/len(pos)
	mean_y = sumy/len(pos)

	new_pos = []
	print "average x", mean_x
	print "average y", mean_y
	for subpos in pos:
		if (abs(mean_x-subpos[0]) < 20000 and abs(subpos[1]-mean_y) < 20000):
			new_pos.append(subpos)
		# else:
		# 	print subpos[0]
	return new_pos



def Kmeans_cluster(X,n_clusters):
    X=np.array(X)
    kmeans=KMeans(n_clusters=n_clusters,random_state=0).fit(X)
    print kmeans.cluster_centers_
    label_id = kmeans.labels_.tolist()
    global center
    center = kmeans.cluster_centers_.tolist()
    # print label_id
    return label_id, center

def sum_up(label_id):
	print "sum_up begin"
	file = open("yellow_tripdata_2016-01.csv", "r")
	reader = csv.reader(file)
	count = 0
	global score
	global score_hour
	for line in reader:
		if (line[1][9]!=str(1) or line[5]==str(0)):
			continue
		if (count > test_num):
			break
		# format
		# 2016-01-01 00:00:00
		# 0123456789
		time_hour = int(line[1][11:13])
		time_min = int(line[1][14:16])
		if (count == len(label_id)):
			break 
		score_hour[label_id[count]][time_hour] += int(line[3])
		# could change to += 1, if passenger_count doesn't have connection with score
		score[label_id[count]][time_hour][time_min] = score[label_id[count]][time_hour][time_min]+1
		count += 1
		# print count, int(line[3])
	# print score, count
	print "sum_up end"

def output():
	gps_file_name = "gps_data.txt"
	gps_file = open(gps_file_name, 'w')
	for i in range(0, node_num):
		for j in range(0, 24):
			for k in range(0, 60):
				s = score[i][j][k]
				center_x = center[i][0] - mean_x
				center_y = center[i][1] - mean_y
				gps_file.write('%d %d %d %d %d %d\n' %(i, j, k, center_x, center_y, s))
	gps_file.close() 


	gps_pos_name = "gps_pos.txt"
	gps_pos_file = open(gps_pos_name, 'w')
	for i in range(0, node_num):
		center_x = center[i][0] - mean_x
		center_y = center[i][1] - mean_y
		gps_pos_file.write('%d %d %d\n' %(i, center_x, center_y))
	gps_pos_file.close()

def plot(flag, pos, label):
	# plot every point
	if flag == 0: 
		for i in pos:
			lat.append(i[0])
			lng.append(i[1])
		print label_id
		plt.scatter(lat,lng,c=label)
	# plot only clusters center 
	else:
		center_x = []
		center_y = []
		color = []
		count = 0
		for i in center:
			center_x.append(i[0])
			center_y.append(i[1])
			color.append(count)
			count += 1
		plt.scatter(center_x, center_y, c=color)
	plt.show()

def plot_scatter():
	index = np.linspace(0, 24, 1440)
	# for i in range (0,24*60):
	# 	index.append(i/60.0)
	values = []
	# for i in range (0, 24):
	# 	for j in range (0, 60):
	# 		values.append(score[0][i][j])
	# 		print score[0][i][j]
	# plt.title("Taxi Call Count")
	# plt.xlabel("Time")
	# plt.ylabel("Number")
	# plt.plot(index, values)
	# plt.show()

	index2 = np.linspace(0, 24, 24)
	values2 = []
	for i in range (0,24):
		values2.append(score_hour[0][i])
	plt.title("Taxi Call Count")
	plt.xlabel("Time")
	plt.ylabel("Number")
	plt.plot(index2, values2)
	plt.show()


def main():
	read_csv()
	pos_res = update_pos(pos)
	label_id, center = Kmeans_cluster(pos_res, n_clusters=node_num)
	print "kmeans finish"
	# plot(0, pos_res, label_id)
	print "plot finish"
	sum_up(label_id)
	print "sum_up finish"
	plot_scatter()
	
	output()


if __name__=='__main__':
    main()
