import numpy
from openpyxl import load_workbook
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

workbook = load_workbook('Lab5 Table.xlsx')
sheets = workbook.get_sheet_names()
sheet = workbook.get_sheet_by_name(sheets[0])

rows = sheet.rows
value = []
for row in rows:
    line = [col.value for col in row]
    value.append(line)

value = value[1:len(value)]
newValue = []
for line in value:
    line = line[1:6]
    newValue.append(line)

x = numpy.array(newValue)

pca = PCA(n_components=1)
newX = pca.fit_transform(x)

print(x)
print(newX)

data = numpy.array(newX)

estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

print(label_pred)
print(centroids)
print(inertia)


K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    meandistortions.append(sum(numpy.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
print(K, meandistortions)

i=0
while i!=8:
    print(meandistortions[i]-meandistortions[i+1])
    i+=1