
# coding: utf-8

# In[ ]:


Carol Calin - 11394846


# In[1]:


# Import packages
import os
import pandas as pd
import numpy as np

# Increase max row display in pandas
pd.options.display.max_rows = 30
pd.options.display.max_columns = 30

# Set directory
os.chdir("C:/Users/acali/machine learning/election_data")
# Import data set
df = pd.read_csv("votes.csv")


# In[2]:


# Creating new target variable in df
# Variable 'target' receives value "1" if more than 50% of votes in the county went to Trump, 0 otherwise (meaning Clinton won)
df['target'] = np.where(df['Trump']>=0.5, 1, 0)


# In[3]:


# 3.1 Selecting the target variables
df1 = df[['population2014', 'population2010', 'population_change', 'POP010210', 'AGE135214', 'age65plus', 'SEX255214', 'White', 'Black','RHI325214', 'RHI425214', 'RHI525214', 'RHI625214', 'Hispanic','RHI825214','POP715213', 'POP645213', 'NonEnglish','Edu_highschool', 'Edu_batchelors', 'VET605213', 'LFE305213', 'HSG010214', 'HSG445213', 'HSG096213', 'HSG495213', 'HSD410213', 'HSD310213', 'Income', 'INC110213', 'Poverty', 'BZA010213', 'BZA110213', 'BZA115213', 'NES010213', 'SBO001207', 'SBO315207', 'SBO115207', 'SBO215207', 'SBO515207', 'SBO415207', 'SBO015207', 'MAN450207', 'WTN220207', 'RTN130207', 'RTN131207', 'AFN120207', 'BPS030214', 'LND110210', 'target' ]]
df1.head(10)


# In[4]:


# Checking the header of the variable to see whether the selection worked
print(df1['target'].value_counts())
print(df1.shape)


# In[5]:


# Preparing for data preprocessing, checking out the standard deviations of the variables
df1.std()


# In[6]:


# 3.2 Create target value

X = df1.iloc[:,:-1]
y = df1.iloc[:,-1]


# In[7]:


print(X.shape)
print(y.shape)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[9]:


from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

pipe = make_pipeline(LinearSVC())
param_grid = {'linearsvc__C':[0.01, 0.1, 1, 10, 100]}

cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=20)

grid_pipe = GridSearchCV(pipe,param_grid, cv = cv)

grid_pipe.fit(X_train,y_train)


# In[10]:


LinearSVC().get_params().keys()


# In[13]:


# Finding the optimal value for the C parameter of the LinearSVC 

print("Best cross-validation accuracy: {:.2f}".format(grid_pipe.best_score_))
print("Test set score: {:.2f}".format(grid_pipe.score(X_test, y_test))) 
print("Best parameters: {}".format(grid_pipe.best_params_))


# In[21]:


linearpipe = make_pipeline(LinearSVC(C=0.01, random_state = 42))
linearpipe.fit(X_train, y_train)

print("No scaler Test score: {:.2f}".format(linearpipe.score(X_test, y_test)))


# Test score: 0.77

# In[15]:


# Pipeline LinearSVC with scalers

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

linearpipe1 = make_pipeline(RobustScaler(),LinearSVC(C=0.01, random_state = 100))
linearpipe1.fit(X_train, y_train)
print("Test score Robust: {:.2f}".format(linearpipe1.score(X_test, y_test)))

linearpipe2 = make_pipeline(StandardScaler(), LinearSVC(C=0.01, random_state = 100))
linearpipe2.fit(X_train, y_train)
print("Test score Standard Scaler: {:.2f}".format(linearpipe2.score(X_test, y_test)))

linearpipe3 = make_pipeline(MinMaxScaler(), LinearSVC(C=0.01, random_state = 100))
linearpipe3.fit(X_train, y_train)
print("Test score Min Max Scaler: {:.2f}".format(linearpipe3.score(X_test, y_test)))


# Score improvement from 0.77 to 0.92 with the Min Max scaler

# In[16]:


# Pipeline LinearSVC with scalers, different C than the one recommended by grid search

linearpipe4 = make_pipeline(RobustScaler(), LinearSVC(C=1000, random_state = 100))
linearpipe4.fit(X_train, y_train)
print("Test score Robust, c = 0.001, not optimal according to gridsearch: {:.2f}".format(linearpipe4.score(X_test, y_test)))

linearpipe5 = make_pipeline(StandardScaler(), LinearSVC(C=1000, random_state = 100))
linearpipe5.fit(X_train, y_train)
print("Test score Standard Scaler, c = 0.001, not optimal according to gridsearch: {:.2f}".format(linearpipe5.score(X_test, y_test)))

linearpipe6 = make_pipeline(MinMaxScaler(), LinearSVC(C=1000, random_state = 100))
linearpipe6.fit(X_train, y_train)
print("Test score Min Max Scaler, c = 0.001, not optimal according to gridsearch: {:.2f}".format(linearpipe6.score(X_test, y_test)))

# Pipeline LinearSVC no scaler, C different than found

linearpipe7 = make_pipeline(LinearSVC(C=1000, random_state = 100))
linearpipe7.fit(X_train, y_train)
print("Test score no scaler and C = 0.001: {:.2f}".format(linearpipe7.score(X_test,y_test)))


# Seems that C indeed has an influence on the accuracy of the models. Both the robust and the standard scaled data is affected, while the min max scaler does not notice any performance drops

# In[17]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedShuffleSplit

pipe = make_pipeline(LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]} 

cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
# Chose the stratified shuffle because it results in stratified randomized folds which retain the sample percentage for each class 

grid = GridSearchCV(pipe, param_grid, cv=cv) 
grid.fit(X_train, y_train) 

# Finding the optimal value for the C parameter of the logistic regression

print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_)) 
print("Test set score: {:.2f}".format(grid.score(X_test, y_test))) 
print("Best parameters: {}".format(grid.best_params_))


# In[18]:


# Pipeline logistic regression no data scaling

pipe1 = make_pipeline(LogisticRegression(C=10, random_state = 1642))
pipe1.fit(X_train, y_train)

print("Test score: {:.2f}".format(pipe1.score(X_test, y_test)))


# In[19]:


# Pipeline logistic regression with scalers, different C than the one recommended by grid search

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

pipe4 = make_pipeline(RobustScaler(), LogisticRegression(C=0.001, random_state = 1642))
pipe4.fit(X_train, y_train)
print("Test score Robust, c = 0.001, not optimal according to gridsearch: {:.2f}".format(pipe4.score(X_test, y_test)))

pipe5 = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, random_state = 1642))
pipe5.fit(X_train, y_train)
print("Test score Standard Scaler, c = 0.001, not optimal according to gridsearch: {:.2f}".format(pipe5.score(X_test, y_test)))

pipe6 = make_pipeline(MinMaxScaler(), LogisticRegression(C=0.001, random_state = 1642))
pipe6.fit(X_train, y_train)
print("Test score Min Max Scaler, c = 0.001, not optimal according to gridsearch: {:.2f}".format(pipe6.score(X_test, y_test)))

pipe7 = make_pipeline(LogisticRegression(C=0.001, random_state = 1642))
pipe7.fit(X_train, y_train)
print("Test score no scaler and C = 0.001: {:.2f}".format(pipe7.score(X_test,y_test)))


# In[20]:


# Pipeline logistic regression with scalers
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

pipe1 = make_pipeline(RobustScaler(), LogisticRegression(C=10, random_state = 1879))
pipe1.fit(X_train, y_train)
print("Test score Robust: {:.2f}".format(pipe1.score(X_test, y_test)))

pipe2 = make_pipeline(StandardScaler(), LogisticRegression(C=10, random_state = 1879))
pipe2.fit(X_train, y_train)
print("Test score Standard Scaler: {:.2f}".format(pipe2.score(X_test, y_test)))

pipe3 = make_pipeline(MinMaxScaler(), LogisticRegression(C=10, random_state = 1879))
pipe3.fit(X_train, y_train)
print("Test score Min Max Scaler: {:.2f}".format(pipe3.score(X_test, y_test)))


# I can see that 
# 1) the base test score of 0.88 is improved if scalers are used on the data, from 0.88 to 0.93 with the Robust scaler, while the best test score is achieved with the Standard and Min Max scalers 0.93
# 2) if I use different values of C than the one found by GridSearchCV, the performance of the Logistic Regression drops, with the lowest score being 0.81

# In[ ]:


Part B


# In[ ]:


Part B


# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


import os
import glob
import pandas as pd

os.chdir("C:/Users/acali/machine learning/ml_data")
train = pd.read_csv("webStats_train.csv", header = None)

test_all = []
for file in glob.glob("webStats_test-*.csv"):
    test_all.append(pd.read_csv(file,  header = None))

test = pd.concat(test_all, axis = 0)
print ("Test shape:", test.shape)
print ("Train shape", train.shape)


# In[ ]:


X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]

print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[ ]:


y_train.dtypes


# In[4]:


pd.to_numeric(y_train)


# In[ ]:


y_train[y_train>2] = 'Very-Popular'


# In[ ]:


y_train[y_train==0] = 'Not-Popular'
y_train[y_train==1] = 'Somewhat-Popular'
y_train[y_train==2] = 'Very-Popular'


# In[ ]:


y_train.head(10)


# In[ ]:


pd.to_numeric(y_test)

y_test[y_test>2] = 'Very-Popular'

y_test[y_test==0] = 'Not-Popular'
y_test[y_test==1] = 'Somewhat-Popular'
y_test[y_test==2] = 'Very-Popular'


# In[ ]:


y_test.head(10)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# In[11]:


# Finding the optimal value for the C parameter of the LinearSVC 

linearsvcpipe = make_pipeline(LinearSVC())
param_grid = {'linearsvc__C':[0.01, 0.1, 1, 10, 100]}

# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=20)

grid_pipe = GridSearchCV(linearsvcpipe,param_grid, cv = 2)
grid_pipe.fit(X_train,y_train)

print("Best cross-validation accuracy: {:.2f}".format(grid_pipe.best_score_))
print("Test set score: {:.2f}".format(grid_pipe.score(X_test, y_test))) 
print("Best parameters: {}".format(grid_pipe.best_params_))


# In[12]:


testpipe1 = make_pipeline(LinearSVC(C=10, random_state = 42))
testpipe1.fit(X_train, y_train)

print("No scaler LinearSVC test accuracy: {:.2f}".format(testpipe1.score(X_test, y_test)))


# In[13]:


# Pipeline LinearSVC with scalers

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

linearpipe1 = make_pipeline(MaxAbsScaler(),LinearSVC(C=10, random_state = 100))
linearpipe1.fit(X_train, y_train)
print("Test score LinearSVC MaxAbsScaler: {:.2f}".format(linearpipe1.score(X_test, y_test)))

linearpipe2 = make_pipeline(StandardScaler(), LinearSVC(C=10, random_state = 100))
linearpipe2.fit(X_train, y_train)
print("Test score LinearSVC Standard Scaler: {:.2f}".format(linearpipe2.score(X_test, y_test)))

linearpipe3 = make_pipeline(MinMaxScaler(), LinearSVC(C=10, random_state = 100))
linearpipe3.fit(X_train, y_train)
print("Test score LinearSVC Min Max Scaler: {:.2f}".format(linearpipe3.score(X_test, y_test)))


# In[ ]:


testpipe2 = make_pipeline(KNeighborsClassifier(n_neighbors=10))
testpipe2.fit(X_train, y_train)

print("No scaler Neighbors test accuracy: {:.2f}".format(testpipe2.score(X_test, y_test)))


# In[ ]:


# Pipelines Neighbors with scalers

neighborpipe1 = make_pipeline(MaxAbsScaler(), KNeighborsClassifier(n_neighbors = 10))
neighborpipe1.fit(X_train, y_train)
print("Test score KNN (10) MaxAbsScaler: {:.2f}".format(neighborpipe1.score(X_test, y_test)))

neighborpipe2 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 10))
neighborpipe2.fit(X_train, y_train)
print("Test score KNN (10) Standard Scaler: {:.2f}".format(neighborpipe2.score(X_test, y_test)))

neighborpipe3 = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors = 10))
neighborpipe3.fit(X_train, y_train)
print("Test score KNN (10) Min Max Scaler: {:.2f}".format(neighborpipe3.score(X_test, y_test)))


# In[16]:


testpipe3 = make_pipeline(RandomForestClassifier(n_estimators=100))
testpipe3.fit(X_train, y_train)

print("No scaler RandomForest (n_estimators = 100) test accuracy: {:.2f}".format(testpipe3.score(X_test, y_test)))


# In[22]:


# Pipelines RandomForestClassifier with scalers

forestpipe1 = make_pipeline(MaxAbsScaler(), RandomForestClassifier(n_estimators = 100))
forestpipe1.fit(X_train, y_train)
print("Test score RandomForest MaxAbsScaler: {:.2f}".format(forestpipe1.score(X_test, y_test)))

forestpipe2 = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 100))
forestpipe2.fit(X_train, y_train)
print("Test score RandomForest Standard Scaler: {:.2f}".format(forestpipe2.score(X_test, y_test)))

forestpipe3 = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators = 100))
forestpipe3.fit(X_train, y_train)
print("Test score RandomForest Min Max Scaler: {:.2f}".format(forestpipe3.score(X_test, y_test)))


# In[22]:


# Pipelines RandomForestClassifier with scalers

forestpipe1 = make_pipeline(MaxAbsScaler(), RandomForestClassifier(n_estimators = 100))
forestpipe1.fit(X_train, y_train)
print("Test score RandomForest MaxAbsScaler: {:.2f}".format(forestpipe1.score(X_test, y_test)))

forestpipe2 = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 100))
forestpipe2.fit(X_train, y_train)
print("Test score RandomForest Standard Scaler: {:.2f}".format(forestpipe2.score(X_test, y_test)))

forestpipe3 = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators = 100))
forestpipe3.fit(X_train, y_train)
print("Test score RandomForest Min Max Scaler: {:.2f}".format(forestpipe3.score(X_test, y_test)))


# Initially, the LinearSVC test accuracy was low, and the scaling of the data more than doubled the test accuracy of the model
# from 0.32 to 0.78 with the MaxAbsScaler and the MinMaxScaler.
# 
# The KNN test accuracy was quite high from the beggining, with a value of 0.76 for the 10 closest neighbors, while scaling the data with any of the methods did not improve the test accuracy but even lowered it by 0.01.
# 
# Finally, the RandomForestClassifier had an impressive test accuracy from the beggining, scoring 0.81. After scaling, no significant change can be observed, except for the MaxAbsScaler which reduced accuracy by 0.01. 

# In[23]:


def add_nan_to_data(train_inputs, miss_prob=0.2):

    """

    Randomly flips a numpy ndarray entry to NaN with a supplied 

    probability.

    

    WARNING: Do not try to add missing values to labels. This is 

    not unsupervised learning.



        :param train_inputs: Numpy ndarray of dims (num_examples, feature_dims)

        :param miss_prob=0.2: Probability that a bit is flipped to NaN. 

    """     

    mask = np.random.choice(2, size=train_inputs.shape, p=[miss_prob,1-miss_prob]).astype(bool)

    input_shape = train_inputs.shape



    #Flatten Inputs and Mask

    train_inputs = train_inputs.ravel()

    mask = mask.ravel()

    train_inputs[~mask] = np.nan

    

    # reshape inputs back to the original shape.

    missing_train_inputs = np.reshape(train_inputs, newshape=input_shape)

    

    return missing_train_inputs



# In[24]:


add_nan_to_data(np.array(X_train), miss_prob = 0.2)


# In[25]:


# Pipeline LinearSVC with missing data and scalers

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer

impute_mean = SimpleImputer(missing_values=np.nan, strategy='mean') 
impute_mean.fit(X_train)


# linearpipe1 = make_pipeline(MaxAbsScaler(),LinearSVC(C=10, random_state = 100))
# linearpipe1.fit(X_train, y_train)
# print("Test score LinearSVC MaxAbsScaler: {:.2f}".format(linearpipe1.score(X_test, y_test)))

# linearpipe2 = make_pipeline(StandardScaler(), LinearSVC(C=10, random_state = 100))
# linearpipe2.fit(X_train, y_train)
# print("Test score LinearSVC Standard Scaler: {:.2f}".format(linearpipe2.score(X_test, y_test)))

# linearpipe3 = make_pipeline(MinMaxScaler(), LinearSVC(C=10, random_state = 100))
# linearpipe3.fit(X_train, y_train)
# print("Test score LinearSVC Min Max Scaler: {:.2f}".format(linearpipe3.score(X_test, y_test)))


# In[ ]:


# sklearn.impute.SimpleImputer continue the assignment

