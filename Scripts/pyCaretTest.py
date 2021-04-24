# compare machine learning algorithms on the sonar classification dataset
from pandas import read_csv
from pycaret.classification import setup
from pycaret.classification import compare_models
from pathlib import Path
import os
import pandas as pd
import time

#C:\Alan\DSC680\Project3Data
start_time = time.time()  # Overall Execution
mypath = "C:/alan/DSC680/"
base_dir = Path(mypath)
appdir = Path(os.path.dirname(base_dir))
backdir = Path(os.path.dirname(appdir))
config_path = base_dir.joinpath('Config')
yaml_filename = "config_anomaly_detection.yaml" # sys.argv[2]
ymlfile = config_path.joinpath(yaml_filename)
project_path = base_dir.joinpath('Project3Data')
data_file = project_path.joinpath('creditcard.csv')
df = pd.read_csv(data_file)


# define the location of the dataset
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
#df = read_csv(url, header=None)


# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# evaluate models and compare models
best = compare_models()
# report the best model
print("best")
print(best)




# tune model hyperparameters on the sonar classification dataset
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from pycaret.classification import setup
from pycaret.classification import tune_model
# define the location of the dataset
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# load the dataset
#df = read_csv(url, header=None)
# set column names as the column number
n_cols = df.shape[1]
df.columns = [str(i) for i in range(n_cols)]
# setup the dataset
grid = setup(data=df, target=df.columns[-1], html=False, silent=True, verbose=False)
# tune model hyperparameters
best = tune_model(ExtraTreesClassifier(), n_iter=200, choose_better=True)
# report the best model
print(best)