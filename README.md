# HEART DISEASE PREDICTION
**Wayne State University**

**CSC5800- Intelligent Systems: Algorithms and Tools**

![image](https://user-images.githubusercontent.com/75294739/231356821-d4b78fe4-3bf8-4c70-925b-38f69986fc23.png)

**Project by:**

Srirekha Dendukuru(hl9411) 

Joshua Rieck(fa9181) 

**Date of submission:**

12/17/2022
 
# Problem Statement

Cardiovascular disease (CVD) is a term that encompasses a range of conditions that affect the heart and blood vessels. These conditions are often caused by the buildup of plaque in the arteries, which can narrow or block the flow of blood to the heart, brain, and other organs. CVD is the leading cause of death globally, accounting for over 17 million deaths each year, according to the World Health Organization. 
In this paper, we have designed several methodologies that will not only allow for the automation of medically diagnosing patients with heart disease, but also enhance patient's value of healthcare and reduce cost. The model performance of these methodologies are evaluated based on the accuracy to correctly identify patients who exhibit heart disease from those who do not.

# Dataset

The dataset used in this project was retrieved from Kaggle. The dataset is an extension of the original, which can be found on the UCI Machine Learning Repository. According to Kaggle, This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. The original data in the study included 76 attributes; for this paper, we will only focus on a sub-set of 14 attributes. For easy implementation into Python, we took this dataset and saved it  out on GitHub. The raw GitHub dataset can be found here.

Contained in this dataset are 1,025 individual instances and the following 14 attributes:

|  |  |
| :------- | :------- |
| Age      | Cholesterol |
| Sex      | Fasting BS |
| Chest pain | Resting ECG |
| Resting Bp | Thalach |
| Exercise angina | Old peak |
| Slope | Ca |
| Thal | Target |


# Language
 
In this project, Python was used to program the models and perform data analysis.  

# Literature Review

There have been many studies done on the heart disease dataset. One of these studies was a paper that tested two classification approaches: Multilayer Perceptron (MLP) and Adaptive Neuro-Fuzzy Inference Systems (ANFIS). The study concluded that the ANFIS classification approach outperformed ANN on the training data, but ANN outperformed ANFIS on the testing data. 

Abushariah, M. , Alqudah, A. , Adwan, O. and Yousef, R. (2014) Automatic Heart Disease Diagnosis System Based on Artificial Neural Network (ANN) and Adaptive Neuro-Fuzzy Inference Systems (ANFIS) Approaches. Journal of Software Engineering and Applications, 7, 1055-1064. doi: 10.4236/jsea.2014.712093.

Another study used association rule mining to investigate the attributes that contribute to heart disease in males and females. The results showed that females had a lower chance of coronary heart disease than males, and that chest pain, resting ECG status, and the presence of exercise-induced angina are potential high-risk factors for heart disease. Alternatively, the study was also able to identify attributes that indicated a "healthy" status for both males and females. 

Jesmin Nahar, Tasadduq Imam, Kevin S. Tickle, Yi-Ping Phoebe Chen, Association rule mining to detect factors which contribute to heart disease in males and females, Expert Systems with Applications, Volume 40, Issue 4, 2013, Pages 1086-1093, ISSN 0957-4174,

Finally, there was another paper that used the  Extreme Learning Machine (ELM) algorithm to predict classifiers on the dataset. The system uses data from the Cleveland Clinic Foundation to model factors such as age, sex, and serum cholesterol levels. The proposed system is intended to provide a warning system for patients with a high probability of heart disease, and has an accuracy of 80% in determining heart disease. The paper suggests that the system could be used to replace costly medical checkups.

S. Ismaeel, A. Miri and D. Chourishi, "Using the Extreme Learning Machine (ELM) technique for heart disease diagnosis," 2015 IEEE Canada International Humanitarian Technology Conference (IHTC2015), 2015, pp. 1-3, doi: 10.1109/IHTC.2015.7238043.
 
# Data Visualization

## Initial Dataset Insights

![image](https://user-images.githubusercontent.com/75294739/231357206-0cccd3a9-905e-4f8b-bd39-b9176d976876.png)

If you look at the above figure, there are more males that are positively diagnosed with heart disease than females.

![image](https://user-images.githubusercontent.com/75294739/231357239-ae822cb8-328d-4550-bc37-9d7d0ba1942d.png)

In the above figure, we created a masked correlation matrix heatmap that compared the relationships of all of our variables against one another. The following observations were identified:
- There appears to be a positive correlation between chest pain scores (cp), maximum heart rate achieved (thalach), and slope of peak exercise segment (slope) on our target variable of heart disease.
- Conversely, there are some negative correlations on our target variable. Those negative correlations are on exercise induced angina (exang), STEMI depression ended by exercise (oldpeak) and the number of major blood vessels colored by florosopy (ca).
 
## Distribution Analysis

 ![image](https://user-images.githubusercontent.com/75294739/231357356-cfeab1b8-24f1-4a35-9fd6-605543044766.png)

While there are many older patients in this dataset, there is a discernible age difference between those who have heart disease and those who do not. Looking at the distribution graph in the above figure, the patients who tend to not have heart disease are older patients. When we look at the box plot of this same variable, the majority of the group falls between late 40’s through early 60’s. We believe the reason for this stark difference between the older cohorts and not having heart disease is due to older patients having more routine heart disease check-ups with their doctors (which leads to older patients being over-sampled in our dataset.)

 ![image](https://user-images.githubusercontent.com/75294739/231357448-54eea504-42c8-4707-9220-950d8e7220be.png)

Another interesting observation is that there is a clear distinction between patients who have heart disease and patients who do not when we look at the oldpeak variable. Looking at the distribution graph in the above figure, patients with low oldpeak scores seem to have a significantly higher peak of positive heart disease diagnosis than those who are negative. Looking at the box plot of this variable, a majority of patients fell between 0 and 2 score, with the overall average falling just under 1.

![image](https://user-images.githubusercontent.com/75294739/231357567-1decccd5-7a5a-46f2-82bd-d9869b46b20f.png)
 
And last, looking at the above figure, the higher a patient’s score on the thalach variable, it was more likely that they would be positively diagnosed with heart disease. The majority of the patients in the dataset fell around 140 to 160 thalach score, with the average being around 150.

## Variable Relationship Analysis

![image](https://user-images.githubusercontent.com/75294739/231357709-2820b475-f477-4030-b625-8e116ddfcd4f.png)
 
In our dataset, there are five variables that have continuous data: age, trestbps, chol, thalach, and oldpeak. Based on the following pairplot grid of these five variables in the above figure, here are some observations:

There appears to be a positive relationship between:
- Trestbps and age
- Oldpeak and age
- Oldpeak and trestbps

There appears to be a negative relationship between:
- Thalach and age
- Oldpeak and thalach
 
# Data Pre-Processing

The 1st pre-processing step we did was to divide the dataset into two separate dataframes: One that contained all 13 of our feature attributes (X) and one that contained our classifier variable (y)

Next, we decided to apply a scaler library on our predictor dataframe. The reason why we did this is because: 
1)	this allows for all of our data points to be more generalized and eliminates 
2)	Helpful in eliminating high variance across different variables
3)	Easier for machine learning algorithms to learn and understand the problem

``` python
X = df.iloc[:,:13] # take the 1st 13 columns of the dataframe
y = df.iloc[: , -1] # take the last column of the dataframe

# scaler = MinMaxScaler() # play around with different scalers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

And last, in order to train our models and test their accuracies, we implemented the train_test_split library to split our two dataframes into a training set and a test set. We decided to put 80% of our data into our training set, the the remaining 20% into our test set.

``` python
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2 , 
                                                 random_state = 1234) 
# splitting the dataset into 80% validation and 20% test
```
 
# Data Mining Algorithms

For this project, we decided to use various eager and lazy methods, to try and accurately predict our classifier variable. All methods that we used are supervised learning models since we already have our data classifiers in both our training and test sets labeled. All algorithms used in this project utilized pre-built packages in the sklearn library:


Eager Learner
- Logistic Regression 
- Naive Bayes
- Decision Tree	

Lazy Learner
- K-Nearest Neighbor

Another technique that we deployed across all of our models was Principal Component Analysis. For this, we reduced the number of features in our dataset from 13 to 2. The reason why we decided to implement this method was to see whether or not simplifying our data points would lead to improved accuracy.

One final technique we used to fine-tune all of our hyper-parameters on our models was implementing the GridSearchCV library. This allowed us to run multiple tests across all of our models using different combinations of the hyper-parameters we wanted to test in order to find the combination that gave us the highest accuracy results.

## Logistic Regression
First step we did was to define all of the hyper-parameters that we wanted to run through the GridSearchCV package. This was done by creating a dictionary variable that contained the name of the hyper-parameter and values we wanted to pass through:

``` python
parameters_lr = {
    'penalty' : ['none' , 'l1' , 'l2'],
    'C'       : [100 , 10 , 1.0 , 0.1 , 0.01],
    'solver'  : ['newton-cg' , 'lbfgs' , 'sag' , 'saga'],
    'max_iter' : [100]
}
```



Next, we defined the number of folds we wanted to implement on the GridSearch:

``` python
cv = 5
```

We then implemented the GridSearch package onto the Logistic Regression package and passed through the hyper-parameters we defined in our dictionary variable above:

``` python
lr = LogisticRegression(random_state = 1234)
clf = GridSearchCV(lr,                   
                   param_grid = parameters_lr,  
                   scoring = 'accuracy',        
                   cv = cv)  

clf.fit(X_train , y_train)
```

Finally, we asked the model to return the hyper-parameter combination that gave us the highest accuracy score:

``` python
print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)
```

### Best Parameters

**Original Data**

The hyper-parameter combination that returned the highest accuracy:
- Penalty: 	none
- C:		100
- Solver: 	newton-cg	

**PCA Data**

The hyper-parameter combination that returned the highest accuracy:
- Penalty: 	l2
- C:		0.01
- Solver:		newton-cg

## Naive Bayes
Since there are fewer hyper-parameters to fine tune on the Gaussian Naive Bayes package, we just ran the package and passed through the training and test datasets to see what the accuracy score is with this method:

``` python
gnb = GaussianNB()
gnb.fit(X_train , y_train)
predict_gnb = gnb.predict(X_test)

print("Accuracy:" , accuracy_score(y_test, predict_gnb))
print('Confusion Matrix:')
print(confusion_matrix(y_test, predict_gnb))
print('Classification Report Table:')
print(classification_report(y_test, predict_gnb))
```

## K-Nearest Neighbor
First step we did was to define all of the hyper-parameters that we wanted to run through the GridSearchCV package. This was done by creating a dictionary variable that contained the name of the hyper-parameter and values we wanted to pass through:

``` python
parameters_knn = {
    'n_neighbors' : [1,3,5,7,9,11,13,15,17], # use odd numbers because we want prediction to select the most frequent classifier
    'weights'       : ['uniform' , 'distance'],
    'metric'  : ['euclidean' , 'manhattan' , 'minkowski']
}
```

Next, we defined the number of folds we wanted to implement on the GridSearch:

``` python
cv = 5
```

We then implemented the GridSearch package onto the K-nearest neighbors package and passed through the hyper-parameters we defined in our dictionary variable above:

``` python
knn = KNeighborsClassifier()
knneighbors = GridSearchCV(knn,                   
                   param_grid = parameters_knn,  
                   scoring = 'accuracy',        
                   cv = cv)  

knneighbors.fit(X_train , y_train)
```

Finally, we asked the model to return the hyper-parameter combination that gave us the highest accuracy score:

``` python
print("Tuned Hyperparameters :", knneighbors.best_params_)
print("Accuracy :",knneighbors.best_score_)
```

### Best Parameters


**Original Data**

The hyper-parameter combination that returned the highest accuracy:
- n_neighbors = 1
- weights = uniform
- metric = manhattan	

**PCA Data**

The hyper-parameter combination that returned the highest accuracy:
- n_neighbors = 17
- weights = distance
- metric = euclidean

## Decision Tree

First step we did was to define all of the hyper-parameters that we wanted to run through the GridSearchCV package. This was done by creating a dictionary variable that contained the name of the hyper-parameter and values we wanted to pass through:

``` python
parameters_dtree = {
    'criterion' : [ 'gini', 'entropy', 'log_loss' ],
    'splitter'  : ['best', 'random'],
    'max_leaf_nodes' : [10,15,20]
}
```

Next, we defined the number of folds we wanted to implement on the GridSearch:

``` python
cv = 5
```

We then implemented the GridSearch package onto the Logistic Regression package and passed through the hyper-parameters we defined in our dictionary variable above:

``` python
dtree = DecisionTreeClassifier()
dtc = GridSearchCV(dtree,                   
                   param_grid = parameters_dtree,  
                   scoring = 'accuracy',        
                   cv = cv)  

dtc.fit(X_train , y_train)
```

Finally, we asked the model to return the hyper-parameter combination that gave us the highest accuracy score:

``` python
print("Tuned Hyperparameters :", dtc.best_params_)
print("Accuracy :",dtc.best_score_)
```

### Best Parameters

**Original Data**

The hyper-parameter combination that returned the highest accuracy:
- criterion = gini
- splitter = best
- max leaf nodes = 20	

**PCA Data**

The hyper-parameter combination that returned the highest accuracy:
- criterion = gini
- splitter = best
- max leaf nodes = 20
 
# Results

## Logistic Regression
### Original Data
When we plugged in our best hyper-parameters into the Logistic Regression model, trained the model on our training data, and tested the accuracy on our test data, we received the following results:

Accuracy: 0.8536585365853658
Confusion Matrix:
[[79 22]
 [ 8 96]]
Classification Report Table:
              precision    recall  f1-score   support

           0       0.91      0.78      0.84       101
           1       0.81      0.92      0.86       104

    accuracy                           0.85       205
   macro avg       0.86      0.85      0.85       205
weighted avg       0.86      0.85      0.85       205

[Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed:    1.0s finished

Here is a diagram of the ROC curve for Logistic Regression:


### PCA Data
When we plugged in our best hyper-parameters into the Logistic Regression model, trained the model on our training data, and tested the accuracy on our test data, we received the following results:

Accuracy: 0.8536585365853658
Confusion Matrix:
[[78 23]
 [ 7 97]]
Classification Report Table:
              precision    recall  f1-score   support

           0       0.92      0.77      0.84       101
           1       0.81      0.93      0.87       104

    accuracy                           0.85       205
   macro avg       0.86      0.85      0.85       205
weighted avg       0.86      0.85      0.85       205

[Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed:    0.4s finished

Here is a diagram of the ROC curve for Logistic Regression:

 
## Naive Bayes
### Original Data
When we trained our data on the Naive Bayes model and tested the accuracy of our test data, we received the following results:

Accuracy: 0.8390243902439024
Confusion Matrix:
[[79 22]
 [11 93]]
Classification Report Table:
              precision    recall  f1-score   support

           0       0.88      0.78      0.83       101
           1       0.81      0.89      0.85       104

    accuracy                           0.84       205
   macro avg       0.84      0.84      0.84       205
weighted avg       0.84      0.84      0.84       205

Here is a diagram of the ROC curve for Naive Bayes:

 
### PCA Data
When we trained our data on the Naive Bayes model and tested the accuracy of our test data, we received the following results:

Accuracy: 0.8585365853658536
Confusion Matrix:
[[78 23]
 [ 6 98]]
Classification Report Table:
              precision    recall  f1-score   support

           0       0.93      0.77      0.84       101
           1       0.81      0.94      0.87       104

    accuracy                           0.86       205
   macro avg       0.87      0.86      0.86       205
weighted avg       0.87      0.86      0.86       205

Here is a diagram of the ROC curve for Naive Bayes:

 
## K-Nearest Neighbor
### Original Data
When we plugged in our best hyper-parameters into the K-Nearest Neighbor model, trained the model on our training data, and tested the accuracy on our test data, we received the following results:

Accuracy: 1.0
Confusion Matrix:
[[101   0]
 [  0 104]]
Classification Report Table:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       101
           1       1.00      1.00      1.00       104

    accuracy                           1.00       205
   macro avg       1.00      1.00      1.00       205
weighted avg       1.00      1.00      1.00       205

Here is a diagram of the ROC curve for K-Nearest Neighbor:

 
### PCA Data
When we plugged in our best hyper-parameters into the K-Nearest Neighbor model, trained the model on our training data, and tested the accuracy on our test data, we received the following results:

Accuracy: 1.0
Confusion Matrix:
[[101   0]
 [  0 104]]
Classification Report Table:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       101
           1       1.00      1.00      1.00       104

    accuracy                           1.00       205
   macro avg       1.00      1.00      1.00       205
weighted avg       1.00      1.00      1.00       205

Here is a diagram of the ROC curve for K-Nearest Neighbor:

 
## Decision Tree
### Original Data
When we plugged in our best hyper-parameters into the Decision Tree model, trained the model on our training data, and tested the accuracy on our test data, we received the following results:

Accuracy: 0.926829268292683
Confusion Matrix:
[[92  9]
 [ 6 98]]
Classification Report Table:
              precision    recall  f1-score   support

           0       0.94      0.91      0.92       101
           1       0.92      0.94      0.93       104

    accuracy                           0.93       205
   macro avg       0.93      0.93      0.93       205
weighted avg       0.93      0.93      0.93       205

Here is a diagram of the ROC curve for Decision Tree:

 
Here is a diagram of our decision tree:

 
### PCA Data
When we plugged in our best hyper-parameters into the Decision Tree model, trained the model on our training data, and tested the accuracy on our test data, we received the following results:

Accuracy: 0.8634146341463415
Confusion Matrix:
[[78 23]
 [ 5 99]]
Classification Report Table:
              precision    recall  f1-score   support

           0       0.94      0.77      0.85       101
           1       0.81      0.95      0.88       104

    accuracy                           0.86       205
   macro avg       0.88      0.86      0.86       205
weighted avg       0.87      0.86      0.86       205

Here is a diagram of the ROC curve for Decision Tree:

 

Here is a diagram of our decision tree:

 





 
# Conclusion

When looking at models performance on the original data and the feature reduced data, the K-nearest neighbors method was the best model for predicting whether or not an individual had heart disease. Another observation is that the feature reduced data did not necessarily produce better results. This could be due to the dataset already having a lower dimensionality. Also, there could be a better n_components amount that we could have selected between 2 and 13 that may have given us a better result, but we decided, for the sake of time, to not test those values out during our PCA reduction analysis.

Regarding the decision trees that we used in our analysis, we initially did not put a threshold on the maximum number of leaf nodes we wanted in our model. This ended up giving us a massive tree that did produce 100% accuracy on both the original data and the feature reduced data as well as 100% ROC score. We decided to implement a maximum 20 leaf nodes on our decision tree just for aesthetic purposes so we could add the diagram to our results section.
 
# Contributions

Literature Review: Rekha
Data Visualization: Rekha
Data Pre-Processing: Josh
Data Mining Algorithms: Josh
Results: Both
Conclusion: Both
