# AI-ML-Project-2023-2024

# Project 1: TRAINS

### Group members:
- Simone Filosofi (284531)
- Simone Angelo Meli (289631)
- Michele Turco (285251)


## INTRODUCTION

In the realm of ThomasTrain, our project revolves around deciphering customer satisfaction without directly soliciting feedback. Armed with the "trains_dataset.csv," our mission is to unravel the factors underlying customer contentment. This endeavor holds immense value for our marketing endeavors as it equips us to pinpoint the elements that keep our patrons delighted. Understanding their satisfaction levels allows us to tailor promotions effectively, ensuring a stronger retention rate and fostering lasting relationships with our clientele.

The dataset provided holds the key to unveiling the subtle cues and patterns that contribute to customer satisfaction. By leveraging this dataset, we aim to uncover hidden insights that might not be overtly expressed. These insights will be pivotal for the marketing team's strategic initiatives, enabling them to craft personalized and engaging campaigns. Ultimately, our goal is to use these findings to create a more enriched experience for our customers, fostering loyalty and bolstering our brand's reputation.

Through meticulous analysis of this dataset, we aim to bridge the gap between customer experiences and tangible insights. Our focus on understanding the unspoken factors influencing satisfaction levels will empower us to make informed decisions. By unraveling these nuances, we aspire to transform this understanding into actionable strategies that resonate with our customers, making their journey with ThomasTrain an exceptional one.


## METHODS

In order to enable readers to better understand the ideas, methods and techniques used in this project, this section will explain the main steps of the project in a more accurate way with respect to the comments that can be found on the main.py file.  
The main python libraries used for this project are:
- pandas: to manipulate the dataset;
- numpy: to perform mathematical operations;
- matplotlib: to plot graphs;
- seaborn: to plot graphs;
- sklearn: to build the models and evaluate their performance;  

The project is divided into 7 main steps, which are described in the following sections.  

### 1) Understanding the dataset
- 1.1)General overview of the dataset: with *'df_trains.head()'* we extracted the first 5 rows of the dataset, to have a general idea of the data we are dealing with;
- 1.2) Showing the dimension of the dataset: We saw that the dataset is composed by 129880 rows and 25 columns, to have a better understanding of the size of the dataset.
- 1.3)Gathering informations from the data: *'df_trains.info()'* allowed us to output the columns' names and data types, and *'df_trains.nunique()'* outputs the number of unique values in each column. In particular, we found out that most of the dtypes were int64(18). We also had the dtype object(6) and float64(1). The object dtype is usually used for strings or where a column contains mixed data types. The float64 dtype is used for floating-point numbers.
- 1.4)Handling missing values: we foud out that 'Arrival Delay in minutes' column has 393 missing values and we substitute these with the mode, even if another option would have been to simply delete the 393 missing values, since they are only a small percentage of the total number of rows.
- 1.5) Data Reduction: assuming they don’t have any predictive power, we removed 'Ticket ID' and 'Date and Time' features. Indeed, as said before, the 'Ticket ID' feature is just an identifier, while the 'Date and Time' feature is not relevant for our analysis, since we are not interested in the time of the day or the day of the week when the customer bought the ticket.

### 2) EDA for feature understanding
As a first step, we had to distinguish categorical and numerical features, since they are treated differently in the EDA. In particular, we found out that there are 5 categorical features and 18 numerical features ("Date and Time"  and "Ticket ID" were removed in the previous step).
- 2.1) **Outliers detection**: outliers are relevant to build our model since they can negatively affect the performance, so as a first step, we  plotted boxplots in order to have a general idea on what is going on. The results are reported in the following plot: <br>  
![BOXPLOT](images\Boxplot.png) <br>  
As we can see, there are many outliers in the 'Arrival Delay in Minutes' and 'Departure Delay in Minutes' features. Even if almost than 75% of the values are less than 10, there are some values that are much higher than the others. In particular, the maximum value for 'Arrival Delay in Minutes' is 1584, while the maximum value for 'Departure Delay in Minutes' is 1592. Another feature that present some outliers is 'Distance', but in this case the outliers are not so relevant, since the maximum value is 4983, which is not so far from the 75% percentile (1359). The other features have a range between 1 and 5, hence the outliers are not so relevant for these features.  
- 2.1) **Descriptive statistics**: this statistics summary, unig *'.describe().T*, only considers numerical features and gives a high-level idea to identify whether the data has any outliers, data entry error, distribution of data such as the data is normally distributed or left/right skewed. The results showed what we already knew from the boxplots, that is the presence of outliers in the 'Arrival Delay in Minutes' and 'Departure Delay in Minutes' features. In addittion, we could see that the average is really low for both the features (15 and 14 minutes). Considering the others numerical features, we can observe a balanced distribution, with the average value that is often not so far from the median value.  
After that, we focused on the categorical features, trying to understand how they are distributed and the most frequent value. The names of the columns are pretty self-explanatory:  
  - Ticket class: The class in which the customers chose to travel. **'Premium'** is the most common value, indicating a potential preference.  
  - Loyaly: whether a customer is loyal or not. A significant proportion of the dataset is marked as **'Loyal'**, which could be indicative of a successful loyalty program or repeated use of the service by the customers.  
  - Gender: There is a slight **female majority** 
  - Work or Leisure: Whether a customer is traveling for work reasons or not. There is an higher number of **work-related travels**.
  - Satisfied: Our target variable. **Not Satisfied** is the most common value in the 'Satisfied' column.  
- 2.2) **Univariate Analysis**: univariate analysis scrutinizes individual variables, exploring their distributions and patterns focusing solely on one variable at a time to understand its behavior and properties within a dataset. At first, we plotted histograms for each numerical feature, in order to understand the distribution of the data. The results are reported in the following plot: <br>  
![HISTOGRAMS](images\Histograms_numerical.png) <br>  
The plots show that customer ratings for onboard services generally skewed high, indicating overall customer satisfaction with the ThomasTrain company's services. In contrast, features like 'Food'n'Drink Rating', 'Seat Comfort Rating', and 'Legroom Service Rating' displayed more diverse customer opinions. Both 'Departure Delay in Minutes' and 'Arrival Delay in Minutes' showed a preponderance of short delays, with occasional longer delays that could significantly impact customer satisfaction. The distribution of 'Distance' suggested that most travels were short, but with enough long-distance trips to merit separate consideration for their impact on satisfaction levels.  
After that, we plotted barplots for each categorical feature, in order to visualize the results previously stated: <br>  
![BARPLOTS](images\Plots_categorical.png)  
<br> 

- 2.3) **Correlation Analysis**: evaluates the relationship between categorical, numerical variables and the target variable, to understand which features were more impactful and correlated with the target variable. The results are reported in the following correlation heatmap: <br>  
![HEATMAP](images\Heatmap_Correlation.png)

### 3) Feature selection:
since there were many features that are not really correlated with the target variable, we drop them, since they are not relevant to train the model.

### 4) Praparing data for modeling
- 4.1) Encoding Categorical Variables:we convert categorical variables into a format suitable for modeling;
- 4.2) Removing Outliers: we used the two-steps approach to find and remove the outliers;
- 4.3) Data Splitting: we generated a training set containing 75% of the observations and a test set containing the remaining 25%.
- 4.4) Distribution of the target variable in the different sets: The ratio between the two classes is the same in both the training and the test set. This is a good thing, since it means that the model will be trained on a balanced dataset and will be able to generalize well. In addittion, we do not have to deal with stratification (splitting the dataset mantaining a balanced ratio between the two classes);
- 4.5) Feature Scaling (Fit and Transform)  we apply fit_transform on the training set and transform on the test set in order to standardize the data.
- 4.6) Creating a Validation Set: the main reason for creating a separate validation set is to have an unbiased evaluation of a model fit on the training dataset;

### 5) Model Building
- Model selection: for our classification task we chose Logistic Regression, Decision Tree, and Random Forest;
- 5.1) Testing Different Models: to have an overview before the tuning, we compute the training and the validation accuracy for all the three models;
- 5.2) Hyperparameter Tuning Using Cross-Validation: as a first step, we searched the best hyperparameters for the model using the RandomizedSearchCV function (this allowed us to explicitly control the number of parameter combinations that are attempted);

### 6) Plotting Learning Curves:
Learning curves illustrate how a model's performance evolves as it's trained on varying amounts of data, revealing insights into overfitting, underfitting, and the impact of dataset size on model accuracy.

### 7) Models Evaluation
- 7.1) Classification metrics: quantitative measures (such as accuracy, precision, recall, F1-score, ROC-AUC) we used to assess the performance of our classification models, providing insights into the model's ability to predict classes accurately, detect true positives, and minimize false predictions. 

- 7.2) Confusion Matrices:  a tabular representation to visualize the performance of a classification algorithm, allowing a clear understanding of true positives, true negatives, false positives, and false negatives. This matrices are fundamental for evaluating a model's precision, recall, accuracy, and other classification metrics.

- 7.3) ROC Curves: (Receiver Operating Characteristic) are graphical representations that illustrate a classification model's performance across various thresholds. They plot the true positive rate (sensitivity) against the false positive rate (1-specificity) for different threshold values, providing a comprehensive overview of a model's ability to distinguish between classes: the area under the ROC curve (AUC-ROC) quantifies the model's overall performance, with a higher AUC indicating better discriminatory power.

- 7.4) Models Comparison:  We give final thoughts about the three classification models we chose, selecting the most suitable model based on the previous results we found and its predictive capabilities.


## EXPERIMENTAL DESIGN

### Choice of Models and Baseline Development: 

- *Logistic Regression*: picked due to its computational efficiency and explanatory power, acting as a fundamental reference point.
- *Decision Trees*:sSelected for their capacity to depict non-linear associations and interpretability, negating the necessity for feature scaling.
- *Random Forest*: an amalgamation of Decision Trees intended to boost efficiency and steadiness, diminishing the likelihood of overfitting while adeptly handling diverse attributes and interactions.

### 1) Outliers Handling

**Main Purpose:**
The primary objective of this experiment was to determine the impact of noisy data on the performance of the machine learning model. As shown in the EDA, the dataset contains outliers that may negatively affect the model's ability to generalize, especially for the feautures 'Arrival Delay in Minutes' and 'Departure Delay in Minutes'. On the other hand, many features have a range between 1 and 5, hence the outliers are not so relevant for these features.
The experiment aims to quantify the impact of outliers on the different model's performance and determine the optimal approach to handling outliers.

**Baseline(s):**
The baseline for comparison is the model's performance without any outlier handling. The baseline is used to determine whether the outlier handling approach improves the model's performance.

**Evaluation Metric(s):**
Since we do not have a specific aim (for example, minimize the false positives or the false negatives), we decided to consider multiple metrics to evaluate the model's performance. In particular, we considered the accuracy, the precision, the recall, the F1-score and the ROC-AUC score. However, for this kind of problem we assumed that minimize false positives is more important than minimize false negatives, since it is better to predict a customer as satisfied and then discover that he is not, than predict a customer as unsatisfied and then discover that he is satisfied. For this reason, we decided to consider the precision as the main metric to evaluate the model's performance.

### Conclusion:
The best approach to handle outliers is to remove them. In fact, the model's performance is better when the outliers are removed.

### 2) Hyperparameter Tuning

**Main Purpose:**
The primary objective of this experiment was to determine the impact of hyperparameter tuning on the performance of the machine learning model. The experiment aims to quantify the impact of hyperparameter tuning on the different model's performance and determine the optimal approach to hyperparameter tuning.

**Baseline(s):**
The baseline for comparison is the model's performance without any hyperparameter tuning. The baseline is used to determine whether the hyperparameter tuning approach improves the model's performance.

**Evaluation Metric(s):** 
For the same reason explained in the previous experiment, we decided to consider multiple metrics to evaluate the model's performance, giving more importance to the precision.

### Conclusion:
The best hyperparameters for each model are described in the main notebook. The best approach to hyperparameter tuning is to use the RandomizedSearchCV function to find a good set of hyperparameters, and then use the GridSearchCV function to find the best hyperparameters in a smaller range of values (local maximum).

### 3) 

#### Evaluation Metrics:

- *Accuracy*: Evaluated the models' comprehensive performance.
- *Precision*: Crucial in reducing false positives in predicting customer contentment.
- *Recall*: Vital in accurately identifying all dissatisfied instances.
- *F1-Score*: Offered a balanced measure encompassing precision and recall, particularly critical in an uneven dataset.
- *ROC-AUC Score*: Assessed the models' capacity to differentiate contented and discontented customers.

All these metrics contributed significantly to a comprehensive model assessment.

## RESULTS

After the hyperparameter tuning, we evaluated the models' performance using the metrics described above. The results are summarized in the following plot:  <br>  
![GETTING STARTED](/images/Comparisson_Classification_Metrics.png)  
<br>For sake of completness, the actual results are reported in the following table:<br>

![GETTING STARTED](images/Metrics_Results.png)  
<br> As we can see, each metrics shows that the Random Forest model has the best performance, even if the differences with the Decision Tree model are really small. In order to have a better understanding of the models' performance that is also easy to interpret, we plotted the confusion matrices for each model: <br>  
![GETTING STARTED](images/Confusion_Matrices.png)  
<br> The confusion matrix plots on the y axis the actual value of the target variable for a given set of features and on the x axis the predicted value. It shows the number of true positives (1 on the y axis and 1 on the x axis), true negatives (0 on the y axis and 0 on the x axis), false positives (0 on the y axis and 1 on the x axis) and false negatives (1 on the y axis and 0 on the x axis). In particular, we can observe that the Random Forest model has the best performance, since it has the highest number of true positives and true negatives, and the lowest number of false positives and false negatives.  
Another relevant metric to evaluate the model's performance is the ROC-AUC score, which is a measure of the model's ability to distinguish between classes. The ROC-AUC score is computed by plotting the true positive rate (sensitivity) against the false positive rate (1-specificity) for different threshold values. The area under the ROC curve (AUC-ROC) quantifies the model's overall performance, with a higher AUC indicating better discriminatory power. The ROC curves for each model are reported in the following plot:  <br>  
![GETTING STARTED](/images/ROC_Curves.png)



## CONCLUSIONS

## 1) General considerations
In general, the dataset seems to be consistent and well structured, since the models built performed well even on the test set (unseen data), also with great results in the last two models. This is a key factor for the project because each conclusion that could be drawn from the analysis of the dataset is reliable and can be used to improve the customer satisfaction.

## 2) Model comparison
The model with the worst result is linear regression, and we can infere it both by observing the metrics and the confusion matrix. The result is not suprising, since it is highly probable that the dataset is not completely linearly separable. The others two model analyzed, Random Forest and Decision Tree, have similar results, but the Random Forest model is slightly better. In fact, the Random Forest model has a better accuracy, precision, recall and F1-score, and a slightly better ROC-AUC score. However, the computational time of the Random Forest model is much higher than the Decision Tree model, so if we want to train the model on a larger dataset, maybe the Decision Tree model is preferable, especially considering the fact that the differences in the performance of the two models are not so relevant, however, in that case the hyperparameter tuning is even more important to avoid the high risk of overfitting.

## 3) Analysis of the most important features
Since we wanted to actually understand the reasons behind the customer satisfaction, we decided to analyze the most important features for each model, assuming that one of the goal of the project is to improve the overall customer satisfaction and hence to understand in which area improvements are needed the most.  
For random forest and decision tree, feature importance scores represent the contribution of each feature to predictive performance, with a range typically between 0 and 1. In the case of logistic regression, the coefficients indicate the strength and direction of feature influence, with an unbounded range.
The most important features for decision tree and random forests are reported in the following table: <br>
![GETTING STARTED](/images/Features_random_forests.png)<br>  
Since the coefficients for the logistic regression are scaled differently from the other two models, we decided to plot the coefficients for the logistic regression model in a separate plot: <br>
![GETTING STARTED](/images/Coefficients_logistic_regression.png)<br>  

