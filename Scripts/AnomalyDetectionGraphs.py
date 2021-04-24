"""
Author: Alan Danque
Date:   20210323
Purpose:Telecom Anomaly Detection Visualizations

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

"""
    Class
V4	0.05147638
V11	0.049106663
V2	0.041691864
V21	0.028938386

"""

df = data_df
df.plot(kind='scatter',x='V4',y='V11', c='Class', cmap='jet')
img_file = results_path.joinpath('Scatterplot_V4_V11_Anomaly_Detection.png')
plt.title('Scatterplot V4 & V11 Anomaly_Detection')
plt.savefig(img_file)
plt.show()

df.plot(kind='scatter',x='V2',y='V21', c='Class', cmap='jet')
img_file = results_path.joinpath('Scatterplot_V2_V21_Anomaly_Detection.png')
plt.title('Scatterplot V2 & V21 Anomaly_Detection')
plt.savefig(img_file)
plt.show()

df.plot(kind='scatter',x='V2',y='V4', c='Class', cmap='jet')
img_file = results_path.joinpath('Scatterplot_V2_V4_Anomaly_Detection.png')
plt.title('Scatterplot V2 & V4 Anomaly_Detection')
plt.savefig(img_file)
plt.show()

df.plot(kind='scatter',x='V4',y='V21', c='Class', cmap='jet')
img_file = results_path.joinpath('Scatterplot_V4_V21_Anomaly_Detection.png')
plt.title('Scatterplot V4 & V21 Anomaly_Detection')
plt.savefig(img_file)
plt.show()

df.plot(kind='scatter',x='V11',y='V21', c='Class', cmap='jet')
img_file = results_path.joinpath('Scatterplot_V11_V21_Anomaly_Detection.png')
plt.title('Scatterplot V11 & V21 Anomaly_Detection')
plt.savefig(img_file)
plt.show()


# draw a boxplot to check for outliers
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
fig.suptitle("Boxplot of 'V4' and 'V11'")
boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
sns.boxplot(y=data_df['V4'], x= data_df['Class'], data=data_df, orient='v', color='#488ab5', ax=ax[0],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[0].set_facecolor('#f5f5f5')
ax[0].set_yticks([0, 10, 20])

sns.boxplot(y=data_df['V11'], x= data_df['Class'], data=data_df, orient='v', color='#488ab5', ax=ax[1],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[1].set_facecolor('#f5f5f5')
ax[1].set_yticks([0, 5, 10])
plt.tight_layout(pad=4.0);
img_file = results_path.joinpath('BoxPlot_V4_V11_Anomaly_Detection.png')
plt.savefig(img_file)
plt.show()



# draw a boxplot to check for outliers
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
fig.suptitle("Boxplot of 'V2' and 'V21'")
boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
sns.boxplot(y=data_df['V2'], x= data_df['Class'], data=data_df, orient='v', color='#488ab5', ax=ax[0],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[0].set_facecolor('#f5f5f5')
ax[0].set_yticks([-90, -30, 30])

sns.boxplot(y=data_df['V21'], x= data_df['Class'], data=data_df, orient='v', color='#488ab5', ax=ax[1],
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)
ax[1].set_facecolor('#f5f5f5')
ax[1].set_yticks([-40, 20, 40])
plt.tight_layout(pad=4.0);
img_file = results_path.joinpath('BoxPlot_V2_V21_Anomaly_Detection.png')
plt.savefig(img_file)
plt.show()


# stacked histogram
import matplotlib.pyplot as plt
f = plt.figure(figsize=(7,5))
ax = f.add_subplot(1,1,1)
