# CIS_508
This repository contains the assignments using machine learning algorithms like Decision Tree classifier, Random Forest classifier,Multilayer Perceptron, Support Vector Machines, K-Nearest Neighbors etc.

Assignment 1

Download the training and test sets from the Kaggle site and train and validate a decision tree classification model from the Scikit-learn machine learning library.
You can change the default parameters of the decision tree classifier to explore other solutions. Explore at least three other solutions by changing the parameters of the decision tree classifier, such as depth of tree, splitting criterion and maximum number of leaf nodes. Use screenshots of Python code to explain what parameters you changed and why.


Kaggle link: https://www.kaggle.com/c/santander-customer-satisfaction/overview

Project Explanation: https://voicethread.com/myvoice/thread/15369932/96050676


Assignment 2

Use Python Scikit Learn to solve both the fraud detection problem and the target marketing problems. Use the following methods in Scikit Learn to solve the problems: (1) Decision Tree classifier and (2) Random Forest classifier

Problem 1- Fraud Detection
Abstract 
A large number of problems in data mining are related to fraud detection. Fraud is a common problem in auto insurance claims, health insurance claims, credit card transactions, financial transaction and so on. The data in this particular case comes from an actual auto insurance company. Each record represents an insurance claim. The last column in the table tells you whether the claim was fraudulent or not. A number of people have used this dataset and here are some observations from them: 
•	“This is an interesting data because the rules that most tools are coming up with do not make any intuitive sense. I think a lot of the tools are overfitting the data set.” 
•	“The other systems are producing low error rates but the rules generated make no sense.” 
•	“It is OK to have a higher overall error rate with simple human understandable rules for a 
business use case like this.” There are two datasets (Excel Files) – 
1. Insurance Fraud – TRAIN-3000, and 
2. Insurance Fraud – TEST-12900. 
Attribute Information: 

Input variables: 
1.	MONTH: Jan through Dec. 
2.	WEEKOFMONTH: Continuous – 1 through 5. 
3.	DAYOFWEEK: Monday through Sunday. 
4.	MAKE: Acura, BMW, Chevrolet, Dodge, Ford, Toyota, VW, Nissan, etc. 
5.	ACCIDENTAREA: Urban, Rural. 
6.	DAYOFWEEKCLAIMED: Monday through Friday. 
7.	MONTHCLAIMED: Jan through Dec. 
8.	WEEKOFMONTHCLAIMED: Continuous – 1 through 5. 
9.	SEX: Male/Female. 
10.	MARITALSTATUS: Married, Single, Divorced, Widow. 
11.	AGE: continuous – 0 through 80. 
12.	FAULT: Policy_Holder, Third_Party. 
13.	POLICYTYPE: Sport-Collision, Sedan-All_Perils, Sedan-Collision, Sedan-Liability etc. 
14.	VEHICLECATEGORY: Sport, Sedan, Utility, etc. 
15.	VEHICLEPRICE: 20000_to_29000,30000_to_39000, 40000_to_59000 etc. 
16.	REPNUMBER: Continuous – 1 through 16 
17.	DEDUCTIBLE: Continuous – 300 through 700. 
18.	DRIVERRATING: Continuous – 1 through 4. 
19.	DAYS_POLICY_ACCIDENT: none, 1_to_7, 8_to_15, 15_to_30, more_than_30, etc. 
20.	DAYS_POLICY_CLAIM: none, 1_to_7, 8_to_15, 15_to_30, more_than_30, etc.
21.	PASTNUMBEROFCLAIMS: none, 2_to_4, more_than_4, etc.
22.	AGEOFVEHICLE: new, 3_years, 4_years, 5_years, 6_years, 7_years, more_than_7, etc.
23.	AGEOFPOLICYHOLDER: 16_to_17, 21_to_25, 31_to_35, etc. 
24.	POLICEREPORTFILED: Yes/No.
25.	 WITNESSPRESENT: Yes/No.
26.	AGENTTYPE: Internal/External.
27.	 NUMBEROFSUPPLIMENTS: none, 1_to_2, 3_to_5, more_than_5.
28.	ADDRESSCHANGE_CLAIM: no_change, under_6_months, 1_year, 2_to_3_years, etc.
29.	NUMBEROFCARS: 1_vehicle, 2_vehicles, 3_to_4 etc. 
30.	YEAR: Continuous – 1994 through 1996.
31.	BASEPOLICY: Collison, All_Perils, Liability etc. 
Output variable (desired target): 
32.	FRAUDFOUND: Yes/No. 

Problem 2- Target Marketing

Abstract

There are two main approaches to enterprise marketing: (1) mass campaigns, targeting several general customers, or (2) directed marketing, targeting a specific set of customers. In this competitive world, the mass campaign strategy is not very productive. Nevertheless, there are challenges to directed marketing: finding potential customers is not very easy, although data mining (DM) techniques are providing some assistance in that regard. The given dataset is from a Portuguese banking institution and was used in their direct marketing campaign to sell term deposits to their customers. You can think of it as cross-selling. The campaign was mostly based on phone calls and the dataset stores general information about customers, details of contacts made with them and the output variable ywhich indicates whether a term deposit was subscribed to by the customer or not. 

There are two datasets –1. PortugueseBank Data –TRAIN, and 2. PortugueseBank Data -TEST 

Attribute Information:
Input variables: # bank client data: 
1 -age(numeric) 
2 -job: type of job (categorical: 'admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services') 
3 -marital: marital status (categorical: 'married', 'divorced', 'single'; note: 'divorced' means divorced or widowed) 
4 -education(categorical: 'unknown', 'secondary', 'primary', 'tertiary') 
5 -default: has credit default? (binary/flag: 'yes', 'no') 
6 -balance: average yearly balance, in euros (numeric) 
7 -housing: has housing loan? (binary/flag: 'yes', 'no') 
8 -loan: has personal loan? (binary/flag: 'yes', 'no') # related with the last contact of the current campaign:
9 -contact: contact communication type (categorical: 'unknown', 'telephone', 'cellular') 
10 -day: last contact day of the month (numeric) 
11 -month: last contact month of the year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec') 
12 -duration: last contact duration, in seconds (numeric)  
13 -campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact) 
14 -pdays: number of days since the last contact with the client from a previous marketing campaign (numeric, -1 means client was not previously contacted) 
15 -previous: number of contacts performed before this campaign and for this client (numeric) 
16 -poutcome: outcome of the previous marketing campaign (categorical: 'unknown', 'other', 'failure', 'success') Output variable (desired target): 
17 -y-has the client subscribed to a term deposit? (binary/flag: 'yes', 'no')

Source:[Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference -ESM'2011, pp. 117-121, GuimarÃ£es, Portugal, October, 2011. EUROSIS.

Links:Article -http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Project Explanation: https://voicethread.com/myvoice/thread/15532495/97051568

Assignment 3

Kaggle Competition Problem: The Homesite competition is to predict the probability that a customer would buy a quoted insurance plan. It's a binary classification problem. The competition, with a prize of $20,000, ended four years ago. The problem description, data and other details are available at:
https://www.kaggle.com/c/homesite-quote-conversion
Classification methods to try: Use the following methods from Scikit Learn to predict the probability that a customer would buy the quoted insurance plan: (1) Multilayer Perceptron, (2) Support Vector Machines (3) Decision Tree (4) Random Forest and (5) K-Nearest Neighbors.

To Do List:
(1) Experiment with SMOTE (or its variations) using different percentages to get a higher accuracy on minority class prediction.
(2) Perform ensemble predictions (one-layer stacking) by combining predictions from the various algorithms. For stacking, try at least five different models - e.g. decision tree, random forest, support vector machines, multilayer perceptron and K-nearest neighbors.
(3) In addition, perform hyperparameter tuning on the stacked model. You can do hyperparameter tuning on individual models if you want, but that is not necessary.
(4) Submit to Kaggle both individual and stacked model predictions and report all such Kaggle scores in an Excel table.
Project Explanation: https://voicethread.com/myvoice/thread/15730664/98386759



Assignment 4

To Do List:
(1) The NLTK library has a number of stemmers, such as the Porter, Lancaster and Snowball Stemmers. Use at least two of those stemmers and compare the differences in some of the stemmed words. Use the word tokenizer to tokenize words before stemming. Select one stemmer for the rest of the analysis.
(2) After stemming, construct the term-document matrix. Eliminate stop words when constructing the term document matrix.
(3) Then construct the TF-IDF matrix from the term-document matrix.
(4) Now combine the TF-IDF matrix with Customer data. Then do one-hot encoding on the categorical variables.
(5) There are two types of feature selection methods - the filter type and the wrapper type. Use both types to determine the best set of features. Use at least two different classification algorithms for feature selection (in both filter and wrapper type). 
•	Filter type - Use Python's SelectKBest module to find the K best features. Use a variety of K values to determine the best set of features for the combined data.
•	Wrapper type - Use the Step Forward Feature Selection method in Python (see reference above) to find the best set of features on the combined data. Use cross-validation.
(6) Split the combined dataset into a training (80%) and a test set (20%). Using the best set of features from each method (filter and wrapper), build new classification models and evaluate them on the test data.
Project Explanation: https://asucarey.voicethread.com/myvoice/thread/15960474/99998835

Assignment 5

Problem 1- Determining Online Shoppers Purchasing Intention 
The problem description, data and other details are available at:
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset# (Links to an external site.) 
A more extensive discussion of the problem is available in this paper:
https://link.springer.com/article/10.1007%2Fs00521-018-3523-0 (Links to an external site.) 

To Do:
(1) This is a classification problem, but you will instead use the dataset to find groupings through clustering. Try to group online shoppers using K-means clustering so that there are distinct groups corresponding to those who intend to buy ("TRUE" in the Target column) vs those who don't intend to buy ("FALSE" in the Target column) . DO NOT USE THE TARGET COLUMN "Revenue" WHEN DOING CLUSTERING. But, using the target values after clustering, compute the percentage of TRUE and FALSE for each group and your overall accuracy of prediction using clustering.
(2) Try manual tuning of hyperparameters to get the best set of clusters.
(3) Using appropriate features, explain/characterize visitors who have the intention to buy ("TRUE" in the Target column).

Problem 2- South German Credit Data
The problem description, data and other details are available at:
https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29 (Links to an external site.) 
A more extensive discussion of the problem is available in this paper:
http://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf (Links to an external site.)  
To Do:
(1) This is also a classification problem, but you will instead use the dataset to find groupings through clustering. Try to group debtors using K-means clustering so that there are distinct groups corresponding to those who are "bad" credits vs those who are "good." DO NOT USE THE TARGET COLUMN "Kredit" WHEN DOING CLUSTERING. But, using the target values after clustering, compute the percentage of good credit ("1" in the target column) vs bad credit ("0" in the target column) for each group and your overall accuracy of prediction using clustering.
(2) Try manual tuning of hyperparameters to get the best set of clusters.
(3) Using appropriate features, explain/characterize debtors who are bad credits ("0" in the Target column).
NOTE: You can open the ASC data file in Excel. But here's a copy since some could not unzip the file.

Project Explanation: https://voicethread.com/myvoice/thread/16128693/101237201


Project 6

To Do List:
(1) First, solve the classification problem using the default mode for all classifiers.
(2) Record accuracy, precision, recall, AUC and F1 score on the test set for each classifier in an Excel table.
(3) Now create an ensemble model (one-layer stacking) by combining predictions from various classifiers. For stacking, use at least three different base classifiers - e.g. decision tree, random forest, gradient boosting. Use random forest, in the default mode, as the stacking classifier.
(4) Add the stacking results on the test set (accuracy, precision, recall, AUC and F1 score) to your Excel table. 
(5) Now do hyperparameter tuning of the random forest stacking classifier by changing at least three different hyperparameters. Use random search for hyperparameter tuning.
(6) In the same Excel table, record accuracy, precision, recall, AUC and F1 score on the test set from the hyperparameter tuned random forest stacking classifier.
(7) Compare all results (accuracy, precision, recall, AUC and F1 score) from different methods - that is, the default base classifiers and the random forest stacking classifiers (both the default and the hyperparameter tuned). Which gave you the best result for each of the metrics accuracy, precision, recall, AUC or F1 score?
(8) If you had more time, explain what you would try next to get better results.

Project Explanation: https://voicethread.com/myvoice/thread/16236787/102021720



