# Red-Wine-Quality
The objective of this project is to predict the wine quality using the wine properties provided in the data set.
#Red Wine Quality

# Source
Paulo Cortez, University of Minho, Guimarães, Portugal, [link](http://www3.dsi.uminho.pt/pcortez) 

A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal @2009

## Data Set
The data set is related to different red wine samples of the Portuguese "Vinho Verde" wine. The quality of wine is scored between 0 (lowest) and 10 (highest). The objective of this project is to predict the wine quality using the wine properties provided in the data set.

Link to Data Set, [winequality-red.csv](http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)

### Attributes
Input variables (based on physicochemical tests):

* 1 - fixed acidity
* 2 - volatile acidity
* 3 - citric acid
* 4 - residual sugar
* 5 - chlorides
* 6 - free sulfur dioxide
* 7 - total sulfur dioxide
* 8 - density
* 9 - pH
* 10 - sulphates
* 11 - alcohol

Output variable (based on sensory data):
12 - quality (score between 0 and 10)

## Flow
1. Data Exploration & Preparation 
2. Use Logistic Regression
3. Random Forest and KNN Classifier
4. Clustering 

## Steps 
### 1. Import data set and libraries 
	* Import the data set and libraries to analyze and visualize the data 
	* Pandas, Numpy, Matplotlib, Seaborn, Sklearn, Statsmodels 

### 2. Explore the data to understand the relationships and properties of the attributes  (ie. mean, std, min/ max value, null values)
	* Explore the data using functions such as data.head(), list (), df.describe() and df.isnull().values.any() to check the the features included in the data set, features, type of values and see if there is any null values 
	* Use sns.countplot to explore the target variable 
	* Use displot, boxplot to visualize the features to understand the their distributions and if the data exists any outliers 

### 3. Clean the dataset using interquartile range method 
	* Quantile the data and remove all the outliers using IQR 
	* Use sns.boxplot() to visualize the data after cleaning 
	* Use sns.pairplot() and correlation plot to understand the relationships between features 

### 4. Partition the data set into train and Test
	* Import train_test_split from sklearn.model selection to split the data 
	* Set 30% of the samples to test and 70% to train and random state to 42

### 5. Using PCA for dimensionality reduction
   	* Standardize the data using StandardScaler()
	* Create pipeline and proceed with PCA 
	* Plot the PCA graph to analyze the PCA features vs. Variance 

### 6. Modelling - Logistic Regression 
	* Perform Logistic Regression as a benchmark using LogisticRegression from sklearn.linear_model

### 7. Advanced Modeling - Random Forest and KNN Classifier 
	* Create Random Forest model using [4,5,10,20,50] for estimators 
	* Use GridSearch to to find the best parameters and analyze all the means as the result of each estimators 
	* Create KNN Classifier model using [3,5,10, 20] for neighbors
	* Use GridSearch to find the best parameters and analyze all the means as the result of each estimators 

### 8. Compare Random Forest and KNN model and select the best model 
	* Compare the mean of the best parameters of both models and select the model that provides the best result 
	* As Random Forest Classifier provided the best result, select the model to test the data set
	* Improve testing result by using binary decisions  

### 9. Cluster Analysis - K-means, DBSCAN, Hierarchical
	* Conduct clustering analysis using method such as Kmeans, DBSCAN and Hierarchical Clustering 


# Watch the Video

* [YouTube Video](https://youtu.be/tawXPaBndiY)

# Links
* [Presentation Video](https://youtu.be/tawXPaBndiY)
* [UCI Machine Learning Repository: Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/wine+quality)

# Group Members
Annie Zhang, Fily Chan, Joelle Leung, Robert Wang
