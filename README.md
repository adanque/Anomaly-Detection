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
- Columns: 	
- Rows:		

 | Feature / Factors | Data Type | Definition | 
 | --------- | --------- | --------- | 
 | State | string | State Abbreviation | 





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

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PCA_Cumulative_Explained_Variance.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Inertia_Elbow_Identification_Plot.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/CreditCard_Anomaly_Detection_Correlation_Matrix.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Principal_Component_Scatter_Plot.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/BoxPlot_V4_V11_Anomaly_Detection.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/BoxPlot_V2_V21_Anomaly_Detection.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V11_V21_Anomaly_Detection.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V4_V21_Anomaly_Detection.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V2_V4_Anomaly_Detection.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V2_V21_Anomaly_Detection.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/Scatterplot_V4_V11_Anomaly_Detection.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/PCA_Heatmap.png?raw=true)

![A remote image](https://github.com/adanque/Anomaly-Detection/blob/main/Results/ModelTesting1.png?raw=true)



### Metrics Evaluation and Observations



### Variable Correlation Reviews




### Principal Component Analysis





### Prediction






### Conclusion: 




## Appendices
### Data Sources
| Source  | Description | URL |
| ------ | ------ | ------ |
| Kaggle | Credit Card Fraud Detection Dataset | https://www.kaggle.com/mlg-ulb/creditcardfraud | 



### References: 






:bowtie:

