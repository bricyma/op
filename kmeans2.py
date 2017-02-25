from sklearn.cluster import KMeans
import numpy as np
import csv
import math as m
import matplotlib.pyplot as plt

pos = []
lat0 = 0
label_id = []
test_num = 400000
node_num = 20
score = [[0]*24 for i in range(node_num)]
lat = []
lng = []

f = 1/298.257722356
a = 6378137.0
e = np.sqrt(f*(2-f))
e = 0.0818191908425
# the origin coordinate of the local navigation frame
lat_ref = 40.7346954346 * np.pi/180
lng_ref = -73.9903717041 * np.pi/180

h_ref = 0
Rn = a/(m.sqrt(1-e**2*(m.sin(lat_ref))**2))
x_ref = (Rn + h_ref) * m.cos(lat_ref) * m.cos(lng_ref)
y_ref = (Rn + h_ref) * m.cos(lat_ref) * m.sin(lng_ref)
z_ref = (Rn * (1-e**2) + h_ref) * m.sin(lat_ref)
A = np.matrix([x_ref, y_ref, z_ref])
R = np.matrix([[-m.sin(lng_ref), m.cos(lng_ref), 0],
               [-m.sin(lat_ref) * m.cos(lng_ref), -m.sin(lat_ref) * m.sin(lng_ref), m.cos(lat_ref)],
               [m.cos(lat_ref) * m.cos(lng_ref), m.cos(lat_ref) * m.sin(lng_ref), m.sin(lat_ref)]])


# LLA -> ECEF
# ECEF -> ENU
def transform_gps(lat, lng, h):

    lat = lat * np.pi/180
    lng = lng * np.pi/180

    # Rn = a/(m.sqrt(1-(e**2)*(m.sin(lat))**2))
    x = (Rn + h) * m.cos(lat) * m.cos(lng)
    y = (Rn + h) * m.cos(lat) * m.sin(lng)
    z = (Rn * (1-e**2) + h) * m.sin(lat)

    gps_ecef_pos = np.matrix([x, y, z])
# x, y, z in local navigation frame
    gps_rel = gps_ecef_pos - A
    gps_navi_pos_t = R * np.transpose(gps_rel)
    gps_navi_pos = np.transpose(gps_navi_pos_t)

    E = gps_navi_pos.item(0)
    N = gps_navi_pos.item(1)
    U = gps_navi_pos.item(2)
    l = [gps_rel.item(0), gps_rel.item(1)]
    return l


def transform(lat, lon):
	er = 6378137.0
	s = np.cos(lat0 * m.pi/180)
	tx = s * er * m.pi *lon / 180.0
	ty = s * er * np.log(np.tan((90.0 + lat) * m.pi / 360.0))
	# tx = lon * math.pi * er / 180.0
	# ty = er * np.log(np.tan((90.0 + lat) * math.pi / 360.0))
	l = [tx,ty]
	return l 	


def read_csv():
	file=open("yellow_tripdata_2016-01.csv", "r")
	reader = csv.reader(file)
	count = 0
	flag = 0
	# pos = []
	for line in reader:
		if (line[1][9]!=str(1) or line[5]==str(0)):
			continue
		# TODO 
		if (count > test_num):
			break
		if (flag!=0):
			lat0 = float(line[5])
			flag = 1
		pos.append(transform_gps(float(line[6]),float(line[5]), 0))

		# pos.append(float[line5])
		# print pos[count]
		count +=1 

def update_pos(pos):
	sumx = 0
	sumy = 0
	for subpos in pos:
		sumx += subpos[0]
		sumy += subpos[1]

	mean_x = sumx/len(pos)
	mean_y = sumy/len(pos)

	pos_copy = pos
	pos = []
	print "average x", mean_x
	print "average y", mean_y
	for subpos in pos_copy:
		if (abs(mean_x-subpos[0]) < 200000 and subpos[1] < -10503986):
			pos.append(subpos)
		else:
			print subpos[0]
	return pos



def Kmeans_cluster(X,n_clusters):
    X=np.array(X)
    kmeans=KMeans(n_clusters=n_clusters,random_state=0).fit(X)
    print kmeans.cluster_centers_
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
		score[label_id[count]][time] += int(line[3])
		count += 1

def output():
	gps_file_name = "gps_data.txt"
	gps_file = open(gps_file_name, 'w')
	for i in range(0, node_num):
		for j in range(0, 24):
			s =  str(score[i][j])
			gps_file.write('%s \n' % s)
	gps_file.close() 

def plot(pos):
	for i in pos:
		lat.append(i[0])
		lng.append(i[1])
	plt.scatter(lat,lng)
	plt.show()

def main():
	read_csv()
	# pos_res = update_pos(pos)
	plot(pos)
	print "plot finish"
	label_id = Kmeans_cluster(pos_res,n_clusters=node_num)
	# print Kmeans_cluster(pos,n_clusters=node_num)
	print "kmeans finish"
	sum_up(label_id)
	print "sum_up finish"
	output()
	print score


if __name__=='__main__':
    main()
