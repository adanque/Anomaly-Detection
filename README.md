# Anomaly-Detection

## _Analyzing Credit Card Transactions to Identify Fraud._

<a href="https://www.linkedin.com/in/alandanque"> Author: Alan Danque </a>

<a href="https://adanque.github.io/">Click here to go back to Portfolio Website </a>

![A remote image](https://adanque.github.io/assets/img/Detection.jpg)

## Abstract: 

Payments using credit cards is one of the most convenient ways to pay for products or services. There are many monetary transactions can be completed all too easily with credit cards. With a simple swipe of a magnetic strip. Insert of a digital chip. Briefly passing a wireless RFID scanner. Voicing one’s credit card numbers over the telephone. Saving the credentials on a browser. A credit card customer can purchase anything. From a small pack of gum from a gas station to airline tickets at the airport. To buying electronic goods from a nearby Target store. With all these convenient ways to pay comes the opportunity for one’s credit card information to be stolen and then used to fraudulently be used to buy items they would normally not buy. According to the author Roman Chuprina, “Unauthorized card operations hit an astonishing amount of 16.7 million victims in 2017. Additionally, as reported by the Federal Trade Commission (FTC), the number of credit card fraud claims in 2017 was 40% higher than the previous year’s number. There were around 13,000 reported cases in California and 8,000 in Florida, which are the largest states per capita for such type of crime. The amount of money at stake will exceed approximately $30 billion by 2020.” (Chuprina, 2021) That is where anomaly detection for credit card fraud can come in. By analyzing the deviation between what is normal and expected behavior, it is possible to identify fraudulent purchases. (Vemula, 2020)

### Project Specific Questions

- Since subscriptions are normally monthly with metrics collected once a month, will it be possible to identify churn in as little as a couple months?
	- Answer: 
	
- What are the indicators that help identify dissatisfaction?
	- Answer: 
	
- What are these factors that lead to loyalty?
	- Answer: 
	
- Is there a way to identify dissatisfaction between monthly subscription payments?
	- Answer: 
	
- Where can this data be derived from?
	- Answer: 
	
- How can we identify how much churn affects the bottom line?
	- Answer: 
	
- Can churn be prevented?
	- Answer: 
	
- Are there indirect factors that lead to churn?
	- Answer: 
	
- Are there an early detection sign?
	- Answer: 
	
- Is there a way to show how much prevented churn has affected the bottom line of cash flow?
	- Answer: 


##  Project Variables / Factors 
### Project Dataset:
- Type:		CSV
- Columns: 	31
- Rows:		284,807

 | Time | Number of seconds between the transactions in the dataset | 
 | -------- | --------- |  
 | V1 | PCA translated | 
 | V2 | PCA translated | 
 | V3 | PCA translated | 
 | V4 | PCA translated | 
 | V5 | PCA translated | 
 | V6 | PCA translated | 
 | V7 | PCA translated | 
 | V8 | PCA translated | 
 | V9 | PCA translated | 
 | V10 | PCA translated | 
 | V11 | PCA translated | 
 | V12 | PCA translated | 
 | V13 | PCA translated | 
 | V14 | PCA translated | 
 | V15 | PCA translated | 
 | V16 | PCA translated | 
 | V17 | PCA translated | 
 | V18 | PCA translated | 
 | V19 | PCA translated | 
 | V20 | PCA translated | 
 | V21 | PCA translated | 
 | V22 | PCA translated | 
 | V23 | PCA translated | 
 | V24 | PCA translated | 
 | V25 | PCA translated | 
 | V26 | PCA translated | 
 | V27 | PCA translated | 
 | V28 | PCA translated | 
 | Amount | Transaction Amount | 
 | Class | 1 for fraud, 0 for normal | 

## Methods used
1.	The Pandas profiling library to assist with generating graphs for exploring the distribution of my data and identify possible fields that need cleaning or removal.
2.	Generate and review performance metrics from my data using AUC score and correlation matrix.
3.	Create filtered sets of my dataset based on the target label and generate correlation matrices of each. List out all the observational differences from the two filtered reviews. And generate supportive plots where possible.
4.	Split my dataset using sklearn model_selection train_test_split or kfold.
5.	Perform principal component analysis to reduce dimensionality of my large dataset and analyze. To Identify components that contribute to my prediction goals.
6.	Use the algorithms: Local outlier factor, One Class SVM, Isolation Forest and Elliptic Envelop to attempt to make predictions.
7.	Review and measure the results of the prediction using a r2 score, mae, mse and rmse.


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
| AnomalyDetectionGraphs.py | Data Visualizations |
| AnomalyDetectionModelPredict.py | Model Evaluation |
| pyCaretTest.py | pyCaret Model Testing |

## Datasets
| File  | Description |
| ------ | ------ |
| creditcard.csv | Credit Card Fraud Dataset | 

## Analyses

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Fraud_Detection_Distribution.png?raw=true)
Imbalanced Churn Dataset

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
Notice the easy to find outliers on the V11 boxplot for Churn class of 1.


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/BoxPlot_V2_V21_Anomaly_Detection.png?raw=true)
The above graph also shows an interesting amount of outlier for component V21.




![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V11_V21_Anomaly_Detection.png?raw=true)
The above scatter plot displays some specific indicator in red - that can help assist with possible churn customers.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V4_V21_Anomaly_Detection.png?raw=true)
The above scatter plot displays some specific indicator that can help assist with possible churn customers.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V2_V4_Anomaly_Detection.png?raw=true)
The above scatter plot does a better job of identifying likely churn customers as noted in the red markers

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V2_V21_Anomaly_Detection.png?raw=true)
The above scatter plot displays some specific indicator that can help assist with possible churn customers.

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V4_V11_Anomaly_Detection.png?raw=true)
The above scatter plot appears to do the best job of identifying likely churn customers. Notice the obvious red grouping outside of those who do not churn in blue.


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PCA_Heatmap.png?raw=true)




### Metrics Evaluation and Observations


![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PrecisionVsRecallAU.png?raw=true)



![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting1.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting2.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting3.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting4.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting5.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting6.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting7.png?raw=true)


### Prediction




### Conclusion: 




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

