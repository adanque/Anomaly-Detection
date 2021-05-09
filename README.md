# Anomaly-Detection

## _Analyzing Credit Card Transactions to Identify Fraud._

<a href="https://www.linkedin.com/in/alandanque"> Author: Alan Danque </a>

<a href="https://adanque.github.io/">Click here to go back to Portfolio Website </a>

<p align="center">
  <img width="460" height="460" src="https://adanque.github.io/assets/img/Detection.jpg">
</p>

## Abstract:  (Note: This analysis project is currently in progress )

Payments using credit cards is one of the most convenient ways to pay for products or services. There are many types of monetary transactions that can be completed easily using credit cards. With a simple swipe of a magnetic strip. Insert of a digital chip. Briefly passing a wireless RFID scanner. Voicing one’s credit card numbers over the telephone. Saving the credentials on a browser. A credit card customer can purchase anything from any vendor in person or online. From a small pack of gum from a gas station to airline tickets at the airport. To buying electronic goods from a nearby Target store. With all these convenient ways to pay - comes the opportunity for one’s credit card information to be stolen. And then used to fraudulently to buy items they would normally not buy. According to the author Roman Chuprina, “Unauthorized card operations hit an astonishing amount of 16.7 million victims in 2017. Additionally, as reported by the Federal Trade Commission (FTC), the number of credit card fraud claims in 2017 was 40% higher than the previous year’s number. There were around 13,000 reported cases in California and 8,000 in Florida, which are the largest states per capita for such type of crime. The amount of money at stake will exceed approximately $30 billion by 2020.” (Chuprina, 2021) That is where anomaly detection for credit card fraud can come in. By analyzing the deviation between what is normal and expected behavior, it is possible to identify fraudulent purchases. (Vemula, 2020)

<p align="center">
  <img width="460" height="460" src="https://adanque.github.io/assets/img/CreditTheif.jpg">
</p>

### Project Specific Questions

- Since the credit card data may likely be streaming in real time, will it be possible to detect credit card fraud?
	- Answer: Yes, given that this project was able to use one period of data it is still able to predict
	
- What visualizations can be used to help identify credit card fraud?
	- Answer: The visualizations that I found are helpful with identifying credit card fraud are boxplots and scatterplots.
	
- What are the factors that can lead to credit card fraud?
	- Answer: Since my variables were PCA translated by the provider of the dataset, I was able to use the resulting data frame export from the correlation matrix to find that my variables V4, V11, V2 and V21 are factors that lead to credit card fraud.
	
- Which algorithms can be used to detect credit card fraud?
	- Answer: After having tested many algorithms, my tests found that ExtraTreesClassifier was the best algorithm for my model.
	
- How many variables can be used to detect credit card fraud?
	- Answer: Using the elbow method with a PCA Cumulative Explained Variance plot, I found that 4 components explain 90% of the variance in my dataset. 
	
- Is it possible to accidentally mistaken a fraudulent credit card charge for a real charge?
	- Answer: Although my test measures per accuracy, recall & precision therefore F1 score are respectively a little higher than 99%, between 72-88% and about 85% there is a low possibility of the model accidentally identifying a normal charge with a  fraudulent charge.
	
- Are there any variables with multicollinearity?
	- Answer: Yes as can seen in the correlation matrix there appears to be multicollinearity with variables V1 & V2, V6 & V7 and V8 , V11 & V12, V21 & V22, V27 & V28.
	
- Can we still create a model if our dataset contains masked variables?
	- Answer: Absolutely, in this project I was able to make accurate predictions - even though the labels for most of the variables were masked. However, the was able to do so since the target response variable was available.	
	
- How can we measure the accuracy of our detection models?
	- Answer: In this project, I was able to use a variety of measures. These include, MAE, MSE, RMSE, r2 Score, F1 Score and perform pipeline validations tests of predicted values.
	
- How accurate will the detection of credit card fraud be?
	- Answer: It can be as accurate as higher than 99%.




##  Project Variables / Factors 
### Project Dataset:
- Type:		CSV
- Columns: 	31
- Rows:		284,807

 | Time | Number of seconds between the transactions in the dataset | 
 | -------- | --------- |  
 | V1 | PCA translated by dataset owner |  
 | V2 | PCA translated by dataset owner |  
 | V3 | PCA translated by dataset owner |  
 | V4 | PCA translated by dataset owner |  
 | V5 | PCA translated by dataset owner |  
 | V6 | PCA translated by dataset owner |  
 | V7 | PCA translated by dataset owner |  
 | V8 | PCA translated by dataset owner |  
 | V9 | PCA translated by dataset owner |  
 | V10 | PCA translated by dataset owner |  
 | V11 | PCA translated by dataset owner |  
 | V12 | PCA translated by dataset owner |  
 | V13 | PCA translated by dataset owner |  
 | V14 | PCA translated by dataset owner |  
 | V15 | PCA translated by dataset owner |  
 | V16 | PCA translated by dataset owner |  
 | V17 | PCA translated by dataset owner |  
 | V18 | PCA translated by dataset owner |  
 | V19 | PCA translated by dataset owner |  
 | V20 | PCA translated by dataset owner |  
 | V21 | PCA translated by dataset owner |  
 | V22 | PCA translated by dataset owner |  
 | V23 | PCA translated by dataset owner |  
 | V24 | PCA translated by dataset owner |  
 | V25 | PCA translated by dataset owner |  
 | V26 | PCA translated by dataset owner |  
 | V27 | PCA translated by dataset owner |  
 | V28 | PCA translated by dataset owner |  
 | Amount | Transaction Amount | 
 | Class | 1 for fraud, 0 for normal | 

## Methods used
1.	I used the Pandas profiling library to assist with generating graphs for exploring the distribution of my data and identify possible fields that need cleaning or removal.
2.	I then generated plots to visualize the distribution of my data, PCA & Inertia plots to understand the grouping, correlation matrix to review relationships of my dataset’s fields, boxplots to analyze the outliers in my dataset and scatter plots to review the distributed spread of my normal and fraudulent classes.
3.	Split my dataset using the sklearn RepeatedStratifiedKFold for model training.
4.	Use the algorithms: Local outlier factor, One Class SVM, Isolation Forest and Elliptic Envelop to attempt to make predictions. Then to further increase specificity over sensitivity, I used pyCaret to help identify the best performing model.
5.	Review and measure the performance of my predictive model using a Precision Recall, F1 Score, MCC, Kapp, r2 score, MAE, MSE, RMSE and the time to train.
6.	Test and verify predictions of my model using test data.
7.	Visualize the Decision Tree of my resulting model using scikit learn’s export_graphviz for model explain ability.


## Pythonic Libraries Used in this project
Package               Version
--------------------- ---------
- imbalanced-learn      0.8.0
- imblearn              0.0
- xgboost               1.3.3
- sklearn               0.0
- pandas                1.2.3
- numpy                 1.20.1
- matplotlib            3.3.4
- pandas-profiling      2.11.0
- pandasql              0.7.3
- pyodbc                4.0.30
- PyYAML                5.4.1
- SQLAlchemy            1.4.2
- seaborn               0.11.1
- pycaret


## Repo Folder Structure

└───Datasets

└───Scripts

└───Results

## Python Files 

| File Name  | Description |
| ------ | ------ |
| AnomalyDetectionEDA.py | Data Reviews, Exploratory Analysis and Data Wrangling |
| AnomalyDetectionVisualizationGraphs.py | Data Visualizations |
| AnomalyDetectionModelEvaluation.py | Model Evaluation |
| AnomalyDetectionPredict.py | Model Prediction |
| pyCaretTest.py | pyCaret Model Testing and Evaluation for the best algorithm|
| AnomalyDetectionModelVisualization.py | Visualization of our Prediction Model using the ExtraTreesClassifier. |

## Datasets
| File  | Description |
| ------ | ------ |
| creditcard.csv | Credit Card Fraud Dataset | 

## Analyses

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Fraud_Detection_Distribution.png?raw=true)

My project dataset is an imbalanced dataset with 492 fraud records out of 284,807 transactions.

### Principal Component Analysis

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PCA_Cumulative_Explained_Variance.png?raw=true)

The above indicates that 4-5 components can explain 90% of data variances.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Inertia_Elbow_Identification_Plot.png?raw=true)

The k means inertia elbow graph above indicates that my dataset can be 
optimally clustered into 4-5 groups.



### Variable Correlation Reviews

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/CreditCard_Anomaly_Detection_Correlation_Matrix.png?raw=true)

I found that if I export the correlation matrix to csv and then sort the feature correlation values by Class.
	


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/CorrelationDataframeOutput.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/CorrelationDataframeOutputSummary.png?raw=true)

I can easily obtain the top 4 correlated features I can use in my predictive model. 
Note: The number 4 was obtained from the PCA Cumulative Explained Variance above.


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Principal_Component_Scatter_Plot.png?raw=true)

Notice the proximity indicating correlations between the clustered features marked in black and one in the red group. This aligns with the 4 features identified earlier per V2, V4, V11 and V21.


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/BoxPlot_V4_V11_Anomaly_Detection.png?raw=true)

Notice the easy to find outliers on the V11 boxplot for fraud class of 1.


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/BoxPlot_V2_V21_Anomaly_Detection.png?raw=true)

The above graph also shows an interesting amount of outlier for component V21.




![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V11_V21_Anomaly_Detection.png?raw=true)

The above scatter plot displays some specific indicator in red - that can help assist with possible fraud customers.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V4_V21_Anomaly_Detection.png?raw=true)

The above scatter plot displays some specific indicator that can help assist with possible fraud customers.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V2_V4_Anomaly_Detection.png?raw=true)

The above scatter plot does a better job of identifying likely fraud customers as noted in the red markers

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V2_V21_Anomaly_Detection.png?raw=true)

The above scatter plot displays some specific indicator that can help assist with possible fraud customers.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V4_V11_Anomaly_Detection.png?raw=true)

The above scatter plot appears to do the best job of identifying likely fraud customers. Notice the obvious red grouping outside of those who do not fraud in blue.



### Metrics Evaluation and Observations

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting1.png?raw=true)

Out of the 5 algorithms, I found that the Isolation Forest algorithm had the best MAE: of .002, MSE: .002 which highlights the accuracy by way of measuring how far our prediction came from the actual values. However, the r2 score for Isolation Forest only came backup with .027 which is very weak to explain variances in our dependent variable due to variability of our independent variables.

The second-best algorithm was LinearRegression per MAE of .003 and MSE of .001. However, the r2 score was moderately good with a .468. Meaning it was better at explaining variances.

I later added an algorithm review using PyCaret to test more algorithms for the best performance.


### PyCaret Model Review

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/pyCaret1.png?raw=true)

PyCaret evaluations between algorithms: Logistic Regression, K Nearest Neighbors, Naïve Bayes and Decision Tree Classifier.  


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting2.png?raw=true)

During the first phases of evaluations we can see K Nearest Neighbors leads in both Accuracy and Precision however, the F1 score that combines an evaluation of both precision and recall is low.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting3.png?raw=true)

As the PyCaret evaluation proceeds, I find it interesting how K Nearest Neighbors hangs in there with high precision while contending with algorithms: Ridge Classifier and Random Forest Classifier. Here we can see that as even more algorithms get added to the evaluation, K Nearest Neighbors continues to hold well to Precision despite not leading accuracy. We can see that the Random Forest Classifier and Ridge Classifier continues to comparatively run well against KNN.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting4.png?raw=true)

Here we can see that KNN is getting surpassed by the algorithms: Random Forest Classifier and Extra Trees Classifier which is much like Random Forest.  With high Accuracy, AUC, Recall and Precision. 

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting5.png?raw=true)

Here we see an evaluation using the F1 to score the relationship of precision and recall, Kappa magnitude score to measure dichotomous agreement with a score higher than .76. And MCC - Mathew’s Correlation Coefficient that measures of .8585. One thing to note though, since we have an imbalanced dataset and F1 score is asymmetric by nature meaning it does not provide similar results if the classes are inverted thus F1 score alone my not be useful as a metric for my project. Another note on the MCC which is a symmetric metric that considers the TP/True Positives, FP/False Positives and FN/False Negatives. It indicates a better score in favor of Extra Trees.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting6.png?raw=true)

Here we can see that the two algorithms: Random Forest Classifier and Extra Trees Classifier are neck and neck in accuracy and very comparable per AUC, Recall and Precision, Kappa and MCC. However, the training time for the Extra Trees Classifier is much better at 9.595 seconds vs 26.728 seconds for Random Forest Classifier.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting7.png?raw=true)

Here my PyCaret evaluation has identified that the best algorithm for my project’s model is “Extra Trees Classifier”. 


### Prediction Pipeline Validation:

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PredictionPipeLine.png?raw=true)

Here we will create a pipeline to feed in a series of values to simulate the input variables of our model and perform our prediction. Each prediction was completed as expected with the correct result using random test data.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Precision_Recall_Curve.png?raw=true)

The above Precision Recall Curve plots how well the prediction vs test values matched per TP/True Positives, FP/False Positives and FN/False Negatives. It was measured using Sklearn’s average_precision_score metric function.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ROC_Curve_Precision_vs_Recall.png?raw=true)

The above Precision vs Recall plot shows how well the prediction vs test values matched per TP/True Positives, FP/False Positives and FN/False Negatives using an area over the curve visualization. It uses Sklearn’s precision_recall_curve metric function.

### Decision Tree Visualization of our new model.



## Full Visualization of the decision tree of my model.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PredictionPipeLineTree.png?raw=true)


## Zoomed in view of the center of the decision tree. 

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PredictionPipeLineTreeZoomed.png?raw=true)





### Conclusion: 

Credit card fraud is an extremely important issue as it devastates a person’s credit, violates their identity and privacy of their personal information. Here we learned that we can detect this type of anomaly to be sensitive to the variations in the features of one’s credit usage. And build a predictive model that has an accuracy greater than 99%.


## Appendices
### Data Sources
| Source  | Description | URL |
| ------ | ------ | ------ |
| Kaggle | Credit Card Fraud Detection Dataset | https://www.kaggle.com/mlg-ulb/creditcardfraud | 



### References: 

Sandberg, E.(August 2020). 15 Disturbing Credit Card Fraud Statistics. Retrieved from: https://www.cardrates.com/advice/credit-card-fraud-statistics/

Bessette, C. (June 2020). How serious a Crime Is Credit Card Theft and Fraud? Retrieved from: https://www.nerdwallet.com/article/credit-cards/credit-card-theft-fraud-serious-crime-penalty

Bank, E. (May 2020). My Credit Card was Used Fraudulently (Here’s What to Do). Retrieved from: https://www.cardrates.com/advice/my-credit-card-was-used-fraudulently-heres-what-to-do/

Li, S. (July 2019). Anomaly Detection for Dummies. Retrieved from: https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1

Aliyev, V. (October 2020). 3 Simple Outlier/Anomaly Detection Algorithms every Data Scientist needs. Retrieved from: https://towardsdatascience.com/3-simple-outlier-anomaly-detection-algorithms-every-data-scientist-needs-e71b1304a932

Alam, M. (September 2020). Anomaly detection with Local Outlier Factor (LOF). Retrieved from: https://towardsdatascience.com/anomaly-detection-with-local-outlier-factor-lof-d91e41df10f2

Flovik, V. (April 2019). Machine learning for anomaly detection and condition monitoring. Retrieved from: https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770

Sucky, R. (October 2020). A Complete Anomaly Detection Algorithm From Scratch in Python: Step by Step Guide. Retrieved from: https://towardsdatascience.com/a-complete-anomaly-detection-algorithm-from-scratch-in-python-step-by-step-guide-e1daf870336e

Garbade, M. (December 2020). How to use Machine Learning for Anomaly Detection and Conditional Monitoring. Retrieved from: https://www.kdnuggets.com/2020/12/machine-learning-anomaly-detection-conditional-monitoring.html#:~:text=The%20main%20goal%20of%20Anomaly,useful%20in%20understanding%20data%20problems

Vemula, A. (May 2020). Anomaly Detection made simple. Credit card fraud case using pycaret package. Retrieved from: https://towardsdatascience.com/anomaly-detection-made-simple-70775c914377

Chuprina, R. (February 2021). Credit Card Fraud Detection: Top ML Solutions in 2021. Retrieved from: https://spd.group/machine-learning/credit-card-fraud-detection/

Alam, M. (October 2020). Machine Learning of anomaly detection: Elliptic Envelope. Retrieved from https://towardsdatascience.com/machine-learning-for-anomaly-detection-elliptic-envelope-2c90528df0a6

Alam, M. (September 2020). Anomaly detection with Local Outlier Factor (LOF). Retrieved from: https://towardsdatascience.com/anomaly-detection-with-local-outlier-factor-lof-d91e41df10f2

Young, A. (November 2020). Isolation Forest is the best Anomaly Detection Algorithm for Big Data Right Now. Retrieved from: https://towardsdatascience.com/isolation-forest-is-the-best-anomaly-detection-algorithm-for-big-data-right-now-e1a18ec0f94f

Dawson, C. (November 2018). Outlier Detection with One-Class SVMS. Retrieved from: https://towardsdatascience.com/outlier-detection-with-one-class-svms-5403a1a1878c






:bowtie:

