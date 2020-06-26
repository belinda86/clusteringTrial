# general
import json
import csv
from matplotlib import pyplot as plt

# nltk for stop words
import nltk
from nltk.corpus import stopwords

# kneed
from kneed import KneeLocator

# sklearn
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import DistanceMetric
from sklearn.cluster import KMeans
from sklearn import metrics

# scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, single
from scipy.spatial.distance import pdist

# data used are obtain from "https://www.kaggle.com/" and is only used for testing purpose
import_filename = "./test_data_2.csv"

# object
class clustering():
  def __init__(self):
    # initialise data_to_compute
    self.load_csv_data()

    # initialise X and extracted features
    self.tokenise_data()

    # write hierarchical_clustered arrays into csv
    self.hierarchical_clustering()

    # write kmean_clustered arrays into csv
    self.kmean_clustering()

  def load_csv_data(self):
    data_array = []
    data_to_compute = []
    key_array = ["id", "date_time", "user_name", "message"]

    with open(import_filename, "r", encoding='utf-8') as read_file:
      csv_reader = csv.reader(read_file, delimiter=',')
      for row in csv_reader:
        tempObj = {}
        for i in range(len(row)):
          tempObj[key_array[i]] = row[i]
          if(key_array[i] == "message"): 
            wordArray = row[i].split(" ")
            for word in wordArray:
              if(word != ""):
                # remove &amp
                first_char = word[0]
                if(first_char == "&"):
                  wordArray.remove(word)

              else:
                wordArray.remove("")
            
            separator = " "
            data_to_compute.append(separator.join(wordArray))

      data_array.append(tempObj)

    self.data_to_compute = data_to_compute

  def load_json_data(self):
    with open("./data.json", "r", encoding='utf-8') as read_file:
      data = json.load(read_file)
    data_to_compute = data["sampleData"]
    self.data_to_compute = data_to_compute

  def tokenise_data(self):
    # english stop words from library
    stop_words = stopwords.words('english')
    
    # known stop words
    stop_words.append("http")
    stop_words.append("https")
    stop_words.append("co")

    # emoji (to import emoji library). To follow-up with library that contain all the emoji code.
    stop_words.append("ðÿ")
    stop_words.append("ðÿœ")
    stop_words.append("ðÿ")
    stop_words.append("¼ðÿ")
    stop_words.append("âœ")
    stop_words.append("ðÿž")
    stop_words.append("œðÿ")
    
    # min freq in docs & ignore words that appear % of the time (max)
    min_df = 5
    max_df = 0.5
    self.max_feature = 30
    vectorizer_count = CountVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, max_features=self.max_feature, lowercase=True)
    count_X = vectorizer_count.fit_transform(self.data_to_compute)
    self.count_list = count_X.toarray().sum(axis=0)
    self.extracted_features = vectorizer_count.get_feature_names()
    # self.extracted_features = dict(zip(extracted_features, count_list))

    vectorizer_tfidf = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, max_features=self.max_feature, lowercase=True)
    self.X = vectorizer_tfidf.fit_transform(self.data_to_compute)

    self.vectorizer_tfidf_list = self.X.toarray().tolist()

  def hierarchical_clustering(self):
    Z = np.array(self.X.toarray())
    linked = linkage(Z, method='ward', metric='euclidean')
    labelList = range(1, len(Z)+1)
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering')
    plt.show()

    dist_array = list(linked[:, 2])
    max_distance = max(dist_array)
    thresh = max_distance / 2
    k = len(set(fcluster(linked, t=thresh, criterion='distance')))

    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster.fit_predict(Z)
    hierarchical_cluster_labels = cluster.labels_
    self.format_and_write_to_csv("./hierarchical_clustered_data.csv", hierarchical_cluster_labels)

  def kmean_clustering(self):
    X = np.array(self.X.toarray())

    # determine k using elbow
    distortions = []
    K = range(1,self.max_feature)
    for k in K:
      kmeanModel = KMeans(n_clusters=k).fit(X)
      distortions.append(kmeanModel.inertia_)

    # Plot elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Value of K')
    plt.ylabel('Sqaured Error (Cost)')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    # k where gradient starts to flatten 
    kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')

    # plot kmeans
    kmeans = KMeans(n_clusters=kn.knee).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
    plt.title('K-mean Clustering')
    plt.xlabel('Document')
    plt.ylabel('Frequency')
    plt.show()

    kmean_label = kmeans.labels_
    self.format_and_write_to_csv("./kmean_clustered_data.csv", kmean_label)

  def format_and_write_to_csv(self, filename, label):
    cluster_array = {}
    for i in range(len(self.data_to_compute)):
      cluster = label[i]
      doc = [self.data_to_compute[i]]
      for score in self.vectorizer_tfidf_list[i]:
        doc.append(score)

      try:
        cluster_array[cluster].append(doc)
      except:
        cluster_array[cluster] = [doc]

    with open(filename, mode='w', encoding='utf-8') as cluster_array_file:
      message = csv.writer(cluster_array_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      message.writerow(["content / extracted features"] + self.extracted_features)
      
      count_array = ["No. of count"]
      for count in self.count_list:
        count_array.append(count)
      message.writerow(count_array)
      message.writerow(["----------------------------------------"])
      message.writerow(["No. of clusters", len(cluster_array)])
      
      for index in cluster_array:
        message.writerow(["----------------------------------------"])
        message.writerow(["No. of message", len(cluster_array[index])])
        for item in cluster_array[index]:
          message.writerow(item)

# main
def main():
  clustering()

main()