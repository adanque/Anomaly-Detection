
"""
Author: Alan Danque
Date:   20210501
Purpose:Visualization of our Prediction Model using the ExtraTreesClassifier.
"""

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import yaml
from pathlib import Path
import time
import os
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.tree import export_graphviz

sttime = time.time()
# C:\Alan\DSC680\Project3Data
start_time = time.time()
mypath = "C:/alan/DSC680/"
base_dir = Path(mypath)
appdir = Path(os.path.dirname(base_dir))
backdir = Path(os.path.dirname(appdir))
config_path = base_dir.joinpath('Config')
yaml_filename = "config_anomaly_detection.yaml"  # sys.argv[2]
ymlfile = config_path.joinpath(yaml_filename)
project_path = base_dir.joinpath('Project3Data')
data_file = project_path.joinpath('creditcard.csv')
results_path = base_dir.joinpath('Project3_Results')
results_path.mkdir(parents=True, exist_ok=True)

analytics_record = results_path.joinpath('generated_analytics_dataset.csv')
anomaly_detection_results = results_path.joinpath('anomaly_detection.csv')
pandasEDA = results_path.joinpath('Anomaly_Detection_PandasProfileReport_output.html')


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
    plt.title('Precision Recall Curve', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    img_file = results_path.joinpath('Precision_Recall_Curve.png')
    plt.savefig(img_file)
    plt.show()


# Read YAML Config
with open(ymlfile, 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
        venvpath = cfg["detection"].get("venvpath")
        ca_certs = cfg["detection"].get("ca_certs")
    except yaml.YAMLError as exc:
        print(exc)

# Read in the Fraud Data
data_df = pd.read_csv(data_file, header=None, skiprows=1)
getdata_header = pd.read_csv(data_file, header=None)
featurelist = getdata_header.iloc[0].values.tolist()
featurelist = featurelist[:-1]
#featurelist = data_df.columns[30:]
print(type(featurelist))
print(featurelist)

# print(data_df.head())
data = data_df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)


# define model to evaluate
model = ExtraTreesClassifier(n_estimators=50)

# Train
model.fit(X, y)
prediction = model.predict(testX)

print("prediction")
print(type(prediction))
print(prediction)
print("testy")
print(type(testy))
print(testy)


# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='ExtraTreesClassifer')
pyplot.title('ROC_Curve Precision vs Recall', fontsize=18)
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
img_file = results_path.joinpath('ROC_Curve_Precision_vs_Recall.png')
pyplot.savefig(img_file)
pyplot.show()

# Using metrics.precision_recall_curve
pr(testy, prediction)

# Extract single tree
estimator = model.estimators_[5]

# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names = featurelist,
                rounded = True, proportion = False,
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

