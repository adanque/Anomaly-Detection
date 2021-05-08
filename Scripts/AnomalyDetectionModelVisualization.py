
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

# define model to evaluate
model = ExtraTreesClassifier(n_estimators=50)

# Train
model.fit(X, y)
# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names = featurelist,
                rounded = True, proportion = False,
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


