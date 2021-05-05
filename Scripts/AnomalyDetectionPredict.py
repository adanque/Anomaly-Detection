"""
Author: Alan Danque
Date:   20210501
Purpose:Prediction using optimal algorithm for Anomaly Detection.
"""

from numpy import mean
from numpy import std
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
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

# Read in the Telecom Churn Data
data_df = pd.read_csv(data_file, header=None, skiprows=1)
# print(data_df.head())
data = data_df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]

"""
# Quick Data Summary Review 
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

# define model to evaluate
# model = KNeighborsClassifier()
model = ExtraTreesClassifier(n_estimators=50)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print("Report Performance before pipeline processing...")
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("Creating pipeline to test ExtraTreesClassifier on dataset")
# scale, then fit model
pipeline = Pipeline(steps=[('s', StandardScaler()), ('m', model)])
# fit the model
pipeline.fit(X, y)
# evaluate on some normal cases (known class 0)
print('Expected Predicted as Normal cases:')
data = [[0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518,
         0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316,
         -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427,
         -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705,
         -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528,
         -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62],
        [11, 1.0693735878819, 0.287722129331455, 0.828612726634281, 2.71252042961718, -0.178398016248009,
         0.337543730282968, -0.0967168617395962, 0.115981735546597, -0.221082566236194, 0.460230444301678,
         -0.773656930526689, 0.32338724546722, -0.0110758870883779, -0.178485175177916, -0.65556427824926,
         -0.19992517131173, 0.1240054151819, -0.980496201537345, -0.982916082135047, -0.153197231044512,
         -0.0368755317335273, 0.0744124028162195, -0.0714074332998586, 0.104743752596029, 0.548264725394119,
         0.104094153162781, 0.0214910583643189, 0.021293311477486,27.5],
        [0, 1.19185711131486, 0.26615071205963, 0.16648011335321, 0.448154078460911, 0.0600176492822243,
         -0.0823608088155687, -0.0788029833323113, 0.0851016549148104, -0.255425128109186, -0.166974414004614,
         1.61272666105479, 1.06523531137287, 0.48909501589608, -0.143772296441519, 0.635558093258208, 0.463917041022171,
         -0.114804663102346, -0.183361270123994, -0.145783041325259, -0.0690831352230203, -0.225775248033138,
         -0.638671952771851, 0.101288021253234, -0.339846475529127, 0.167170404418143, 0.125894532368176,
         -0.00898309914322813, 0.0147241691924927, 2.69],
        [1, -1.35835406159823, -1.34016307473609, 1.77320934263119, 0.379779593034328, -0.503198133318193,
         1.80049938079263, 0.791460956450422, 0.247675786588991, -1.51465432260583, 0.207642865216696,
         0.624501459424895, 0.066083685268831, 0.717292731410831, -0.165945922763554, 2.34586494901581,
         -2.89008319444231, 1.10996937869599, -0.121359313195888, -2.26185709530414, 0.524979725224404,
         0.247998153469754, 0.771679401917229, 0.909412262347719, -0.689280956490685, -0.327641833735251,
         -0.139096571514147, -0.0553527940384261, -0.0597518405929204, 378.66]]

print("Summary of accuracy of expected normal cases.")
for row in data:
    # make prediction
    yhat = pipeline.predict_proba([row])
    # get the probability for the positive class
    result = yhat[0][1]
    print('>Predicted=%.3f (expected 0)' % result)

# evaluate on some fraud cases (known class 1)
print('Expected Predicted as Fraud cases:')
data = [
    [406, -2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468, -0.522187864667764, -1.42654531920595,
     -2.53738730624579, 1.39165724829804, -2.77008927719433, -2.77227214465915, 3.20203320709635, -2.89990738849473,
     -0.595221881324605, -4.28925378244217, 0.389724120274487, -1.14074717980657, -2.83005567450437,
     -0.0168224681808257, 0.416955705037907, 0.126910559061474, 0.517232370861764, -0.0350493686052974,
     -0.465211076182388, 0.320198198514526, 0.0445191674731724, 0.177839798284401, 0.261145002567677,
     -0.143275874698919, 0],
    [472, -3.0435406239976, -3.15730712090228, 1.08846277997285, 2.2886436183814, 1.35980512966107, -1.06482252298131,
     0.325574266158614, -0.0677936531906277, -0.270952836226548, -0.838586564582682, -0.414575448285725,
     -0.503140859566824, 0.676501544635863, -1.69202893305906, 2.00063483909015, 0.666779695901966, 0.599717413841732,
     1.72532100745514, 0.283344830149495, 2.10233879259444, 0.661695924845707, 0.435477208966341, 1.37596574254306,
     -0.293803152734021, 0.279798031841214, -0.145361714815161, -0.252773122530705, 0.0357642251788156,529],
    [7519, 1.23423504613468, 3.0197404207034, -4.30459688479665, 4.73279513041887, 3.62420083055386, -1.35774566315358,
     1.71344498787235, -0.496358487073991, -1.28285782036322, -2.44746925511151, 2.10134386504854, -4.6096283906446,
     1.46437762476188, -6.07933719308005, -0.339237372732577, 2.58185095378146, 6.73938438478335, 3.04249317830411,
     -2.72185312222835, 0.00906083639534526, -0.37906830709218, -0.704181032215427, -0.656804756348389,
     -1.63265295692929, 1.48890144838237, 0.566797273468934, -0.0100162234965625, 0.146792734916988, 1],
    [7526, 0.00843036489558254, 4.13783683497998, -6.24069657194744, 6.6757321631344, 0.768307024571449,
     -3.35305954788994, -1.63173467271809, 0.15461244822474, -2.79589246446281, -6.18789062970647, 5.66439470857116,
     -9.85448482287037, -0.306166658250084, -10.6911962118171, -0.638498192673322, -2.04197379107768, -1.12905587703585,
     0.116452521226364, -1.93466573889727, 0.488378221134715, 0.36451420978479, -0.608057133838703, -0.539527941820093,
     0.128939982991813, 1.48848121006868, 0.50796267782385, 0.735821636119662, 0.513573740679437, 1]
]

print("Summary of accuracy expected fraud predictions.")
for row in data:
    # make prediction
    yhat = pipeline.predict_proba([row])
    # get the probability for the positive class
    result = yhat[0][1]
    print('>Predicted=%.3f (expected 1)' % result)

print("ExtraTreesClassifier Prediction Complete. Duration: --- %s seconds ---" % (time.time() - sttime))
