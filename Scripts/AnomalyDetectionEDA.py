"""
Author: Alan Danque
Date:   20210323
Purpose:Preliminary Pandas Profiling for Telecom Churn Data

rows attributes
( , )
Rows:
Fields:
"""
import imblearn
import seaborn as sns
import os
import json
import pandas as pd
import glob
import time
#from textblob import TextBlob
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandasql import sqldf
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from numpy.core.defchararray import find
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from pandas_profiling import ProfileReport
import string
from io import StringIO
from html.parser import HTMLParser
import yaml
from pathlib import Path
from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
#use_label_encoder =False, eval_metric='logloss' - To suppress warnings.
import sklearn
from sklearn import metrics
print('scikit-learn version: {}'.format(sklearn.__version__))


def pr(y_true, y_scores):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    auc = metrics.auc(recall, precision)
    print('precision: %s' % precision)
    print('recall: %s' % recall)
    print('thresholds: %s' % thresholds)
    print('Naive auPRC: %s' % auc)
    print('average_precision_score auPRC: %s' % metrics.average_precision_score(y_true, y_scores))
    plt.plot(recall, precision);
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall AU', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #xlim(0, 1);
    #ylim(0, 1);
    plt.show()


#C:\Alan\DSC680\Project3Data
start_time = time.time()
mypath = "C:/alan/DSC680/"
base_dir = Path(mypath)
appdir = Path(os.path.dirname(base_dir))
backdir = Path(os.path.dirname(appdir))
config_path = base_dir.joinpath('Config')
yaml_filename = "config_anomaly_detection.yaml" # sys.argv[2]
ymlfile = config_path.joinpath(yaml_filename)
project_path = base_dir.joinpath('Project3Data')
data_file = project_path.joinpath('creditcard.csv')
results_path = base_dir.joinpath('Project3_Results')
results_path.mkdir(parents=True, exist_ok=True)

analytics_record = results_path.joinpath('generated_analytics_dataset.csv')
anomaly_detection_results = results_path.joinpath('anomaly_detection.csv')
pandasEDA = results_path.joinpath('Anomaly_Detection_PandasProfileReport_output.html')

# Read YAML Config
with open(ymlfile, 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
        venvpath = cfg["detection"].get("venvpath")
        ca_certs = cfg["detection"].get("ca_certs")
    except yaml.YAMLError as exc:
        print(exc)

# Read in the Telecom Churn Data
data_df = pd.read_csv(data_file)

print("DataFrame Shape")
print(data_df.shape) # 49,352 obs and 15 vars
print("Columns")
print(data_df.columns)
print("count of NAs")
print(data_df.isna().sum())
print("count of Nulls")
print(data_df.isnull().values.any())
print("summary")
print(data_df.describe())

print("Generate Pandas Profile for Telecom Churn Data")
#bprof = ProfileReport(data_df)
#bprof.to_file(output_file=pandasEDA)

print(data_df['Class'].value_counts())

# EDA

"""
Review these fields pressure on Class
V4
V11
V2
V21
V27

0.05147638
0.049106663
0.041691864
0.028938386
0.023116066


#plot the scatter plot of balance and salary variable in data
plt.scatter(data.salary,data.balance)
plt.show()

#plot the scatter plot of balance and age variable in data
data.plot.scatter(x="age",y="balance")
plt.show()


#plot the pair plot of salary, balance and age in data dataframe.
sns.pairplot(data = data, vars=['salary','balance','age'])
plt.show()

"""






count_class = pd.value_counts(data_df['Class'], sort=True)
count_class.plot(kind='bar',rot = 0)
plt.title('Anomaly Detection Distribution')
plt.xlabel('Fraud')
img_file = results_path.joinpath('Fraud_Detection_Distribution.png')
plt.savefig(img_file)
plt.show()



y_true = np.array([0,0,0,0,0,1])
y_scores = np.array([.5,.5,.5,.5,.5,.5])
pr(y_true, y_scores)





# Variable Correlation Matrix
plt.figure(figsize = (16,16))
corr = data_df.corr(method='kendall')
sns.set(font_scale=.8)
sns.heatmap(corr, vmin=corr.values.min(), vmax=1, fmt='.1f', square=True, cmap="Blues", linewidths=0.1, annot=True, annot_kws={"size":14})
plt.title('Variable Correlation Matrix', fontsize = 18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
img_file = results_path.joinpath('CreditCard_Anomaly_Detection_Correlation_Matrix.png')
plt.savefig(img_file)
plt.show()
img_file = results_path.joinpath('CreditCard_Anomaly_Detection_Correlation_Matrix.csv')
corr.to_csv(img_file)



# PCA
# Separate out the feature response (dependent/prediction) variable
# feature matrix
X = data_df.drop('Class', axis=1)
# target vector
y = data_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

#Importing the PCA module
#from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data
pca.fit(X_train_rus)
PCA(copy=True, iterated_power='auto', n_components=None, random_state=42,
    svd_solver='randomized', tol=0.0, whiten=False)

#print("pca.components_")
#print(pca.components_)

# Note: the features of the original dataset was already PCA transformed to protect the innocent
colnames = list(X.columns)
pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'PC3':pca.components_[2],'Feature':colnames})
pcs_df.head(10)


#Making the screeplot - plotting the cumulative variance against the number of components
# fig = plt.figure(figsize = (12,9))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components', fontsize=10)
plt.ylabel('cumulative explained variance', fontsize=10)
plt.title('PCA Cumulative Explained Variance', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
img_file = results_path.joinpath('PCA_Cumulative_Explained_Variance.png')
plt.savefig(img_file)
plt.show()


# Create a KMeans instance with k clusters: model
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fit model to samples
    model.fit(pcs_df.iloc[:, :3])
    # labels = model.predict(pcs_df.iloc[:, :3])
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
plt.plot(ks, inertias, '-o', color="black")
plt.title('KMeans Inertia for Elbow Method', fontsize = 12)
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
img_file = results_path.joinpath('Inertia_Elbow_Identification_Plot.png')
plt.savefig(img_file)
plt.show()

print("4 Clusters is optimal in our case! Decided by PCA Cumulative Variance and Kmeans Inertia!")


legend_names = ["Cluster 1","Cluster 2","Cluster 3"]
LABEL_COLOR_MAP = {0 : 'r', 1 : 'k', 2 : 'b', 3 : 'g'}
model = KMeans(n_clusters=4)
model.fit(pcs_df.iloc[:,:2])
labels = model.predict(pcs_df.iloc[:,:2])
label_color = [LABEL_COLOR_MAP[l] for l in labels]
print(type(labels))
print(labels)
plt.scatter(pcs_df.PC1, pcs_df.PC2, c=label_color) #, label=legend_names)
plt.xlabel('Principal Component 1', fontsize=10)
plt.ylabel('Principal Component 2', fontsize=10)
for i, txt in enumerate(pcs_df.Feature):
    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
#fig, ax = plt.subplots()
#ax.legend(LABEL_COLOR_MAP, legend_names)
#plt.legend(legend_names, loc='best')
plt.title('Principal Component Analysis', fontsize = 12)
img_file = results_path.joinpath('Principal_Component_Scatter_Plot.png')
plt.savefig(img_file)
plt.show()


# Looks like approx. 50 components are enough to describe 90% of the variance in the dataset
# We'll choose 50 components for our modeling
#Using incremental PCA for efficiency - saves a lot of time on larger datasets
pca_final = IncrementalPCA(n_components=16)
df_train_pca = pca_final.fit_transform(X_train_rus)
print("df_train_pca.shape")
print(df_train_pca.shape)

#Creating correlation matrix for the principal components - I expect little to no correlation
df_corr = data_df.corr()
corrmat = np.corrcoef(df_train_pca.transpose())
plt.figure(figsize = (16,16))
sns.set(font_scale=.8)
sns.heatmap(corrmat, vmin=df_corr.values.min(), vmax=1, fmt='.1f', square=True, cmap="Blues", linewidths=0.1, annot=True, annot_kws={"size":18})
plt.title('PCA Heatmap', fontsize = 16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
img_file = results_path.joinpath('PCA_Heatmap.png')
plt.savefig(img_file)
plt.show()

corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)




data_df = data_df
pysqldf = lambda t: sqldf(t, globals())
t = """
SELECT * FROM data_df a           
        ;"""
ncdf = pysqldf(t)

print("Complete Duration: --- %s seconds ---" % (time.time() - start_time))

