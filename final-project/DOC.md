# Final Project Guide

*Hello, brave warrior! Today we will overcome the last frontier of the **DRU DS Fundamentals** course completing the **Final Project**.*

*The aim of the **Final Project** is to allow you to peek behind the curtain of **production Machine Learning.**  
In this guide, we will go through the process and develop web API, data preprocessing and ML modules. So, let's get started!*

# Table Of Contents
-  [Intro](#intro)
-  [Project Structure](#project-structure)
-  [Feature Engineering & Data Preprocessing](#feature-engineering--data-preprocessing)
    -  [Feature Engineering](#feature-engineering)
    -   [Data Preprocessing](#data-preprocessing)
-  [Model Selection, Training and Saving](#model-selection-training-and-saving)
-  [API](#api)


## Intro

This tutorial is not like our previous guides, because here you'll see the whole development process from the **feature engineering** stage to the **saved model usage** with full code. **The guide will show you all the phases of the Final Project activity to help you with implementation of your own project for solving the chosen task.** You should read it carefully in order to complete the **Final Project** activity without any misunderstandings. 

We will consider such stages as:

- **Feature Engineering & Data Preprocessing**
- **Model Selection, Training and Saving**
- **Creating a small Web API (as bonus)**

By working with the data from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic), we will go through all stages of the ML project development.

**After reading this tutorial your tasks will be:**

- setting **any task you like** to solve it using ML (classification, regression, etc.). You can come up with it by yourself or choose a task from Kaggle. **Don't try to submit the code from this guide :)**
- finding a dataset, which fits your task
- implementing a data preprocessing module
- choosing, training, and saving the model
- writing simple API to use your trained estimator
- dockerizing your app

> **Important note:**
Size of your train set must be from **600** to **20000** examples, size of the validation set must be between **100** and **10000**.

You are allowed to choose any dataset you want.
However, we found **10 datasets, that we've tested and recommend to use** for your project:

1.[https://www.kaggle.com/sakshigoyal7/credit-card-customers](https://www.kaggle.com/sakshigoyal7/credit-card-customers)  
2.[https://www.kaggle.com/jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)  
3.[https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction)  
4.[https://www.kaggle.com/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)  
5.[https://www.kaggle.com/dansbecker/melbourne-housing-snapshot](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot)  
6.[https://www.kaggle.com/CooperUnion/cardataset](https://www.kaggle.com/CooperUnion/cardataset)  
7.[https://www.kaggle.com/marcopale/housing](https://www.kaggle.com/marcopale/housing)  
8.[https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
9.[https://www.kaggle.com/iliassekkaf/computerparts](https://www.kaggle.com/iliassekkaf/computerparts)  
10.[https://www.kaggle.com/harlfoxem/housesalesprediction](https://www.kaggle.com/harlfoxem/housesalesprediction)


## Project Structure
```
app
    ├── data                        - contains train and validation data
    │   ├── train.csv               - train set 
    │   └── val.csv                 - validation set (must contain target values)
    ├── models                      - this folder contains a trained estimator.
    │   └── <name>.pickle           - trained estimator. 
    │
    ├── settings                    - here you can store different constant values, connection parameters, etc.
    │   ├── constants.py            - multiple constants storage for their convenient usage.
    │   └── specifications.json     - specifications of your data preprocessing operations.   
    │   
    ├── utils                       - this folder contains instruments we'll use to work with dataset.
    │   ├── __init__.py             - init file for the package. 
    │   ├── dataloader.py           - dataloader. 
    │   ├── dataset.py              - class dedicated for giving info about the dataset.
    │   ├── predictor.py            - predictor.
    │   └── trainer.py              - train script.
    │ 
    ├── app.py                      - route, app.
    │
    ├── requirements.txt			- list of libraries used for Dockerization 
    │
    └── Dockerfile					- commands used for Dockerization
```
You have to define the same structure for your project

## Feature Engineering & Data Preprocessing

These two stages are usually the **most important** and difficult in the development of Data Science projects and may take up to **80%** of the project development time. To perform data preprocessing, you need to **"feel"** the data - that is understanding the meaning of all features in the dataset, exploring and knowing all relations between them, etc. To learn more about the role of Data Preprocessing, read [**here**](https://hackernoon.com/what-steps-should-one-take-while-doing-data-preprocessing-502c993e1caa). 

>**Important Note:**
>Further we will use train.csv and val.csv, but they are not exactly train.csv and test.csv from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic)
>Both our train.csv and val.csv contains target value "Survived" (but test.csv from competition don't). Take this into account when doing your project
>So just split train.csv from the competition into train.csv and val.csv in a ratio of 80/20 and download them

Now let's get to practice and define some constants:

`settings/constants.py`
```python
import os

DATA_FOLDER = 'data'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(DATA_FOLDER, 'val.csv')
```

Firstly, we need to take a look at the data, so let's load the dataset:
```python
import numpy as np
import pandas as pd
import re as re

from settings.constants import TRAIN_CSV, VAL_CSV 

train = pd.read_csv(TRAIN_CSV, header = 0, dtype={'Age': np.float64})
val  = pd.read_csv(VAL_CSV , header = 0, dtype={'Age': np.float64})
full_data = [train, val]

train.head()
```

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/train_overview.jpg?raw=true">
</div>

The **target value** is the **`Survived`** column and other columns are the **features,** so our task is to **predict** surviving of the passenger depending on his **features**. ****To perform correct **Data Preprocessing,** we need to consider all the **features** and find out their **implications** on the dataset.

### Feature Engineering

**`Pclass:`** There is no missing values in this feature and it's already a numerical value. So let's check its impact on our training set:
```python
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()
```
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/pclass.png?raw=true">
</div>

We can see, that passengers, who traveled first-class had more chances to survive.

**`Sex:`** 
```python
train[["Sex", "Survived"]].groupby(['Sex'], as_index = False).mean()
```

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/sex_survived.jpg?raw=true">
</div>


Women had many more chances to survive on Titanic (seems like James Cameron knew about the statistics ;) )

**`SibSp and Parch:`** Having the number of siblings/spouses and the number of children/parents, we can create a new feature called **Family Size**.
```python
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean()
```
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/sibsp.png?raw=true">
</div>

It seems to have a good effect on our prediction, but let's go further and **categorize** people to check whether they were alone on the ship or not.

```python
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
```
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/isalone.png?raw=true">
</div>

Good! The impact is considerable.

**`Embarked:`** the embarked feature has some missing values.  Let's fill them with the most occurred value ( 'S' ):
```python
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
```
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/embarked.png?raw=true">
</div>

**`Fare:`** Fare also has some missing values and we will replace them with the **median**. Then, we **categorize** it into 4 ranges.
```python
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()
```
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/fare.png?raw=true">
</div>

As we can see, passengers with the cheapest tickets had almost no chance to survive and the probability to survive grew with the price of the ticket.

**`Age:`** we have plenty of missing values in this feature. Let's generate random numbers between (mean - std) and (mean + std) to fill Nans, then we will categorize age into 5 range:
```python
age_avg = train['Age'].mean()
age_std = train['Age'].std()    
age_null_count = train['Age'].isnull().sum()
    
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
train['Age'][np.isnan(train['Age'])] = age_null_random_list
train['Age'] = train['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
```
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/age.png?raw=true">
</div>

We can see that children had the highest chance to survive.

**`Name:`** inside this feature we can find passengers' titles.
```python
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

train['Title'] = train['Name'].apply(get_title)

pd.crosstab(train['Title'], train['Sex'])
```

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/get_title.jpg?raw=true">
</div>      

    
Now that we have titles, let's categorize them and check the title impact on survival rate.
```python
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```
<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/title-cat.png?raw=true">
</div>

As we can judge, young ladies had more chances to survive than other people.

### Data Preprocessing

Let's process the data taking into the account the insights we got from Feature Engineering.

Now we will define processing operations and their descriptions regarding mentioned possible operations, which we will use in `specifications.json`:

**1. columns combinations:**
```python
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
```

**2. replace value**
```python
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
```

**3. fill Nan with mode**
```python
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
```

**4. fill Nan with median**
```python
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
```

**5. binning with qcut**
```python
train['Fare'] = pd.qcut(train['Fare'], 4)
```

**6. fill Nan with values from random distribution**
```python
age_avg = train['Age'].mean()
age_std = train['Age'].std()
age_null_count = train['Age'].isnull().sum()
rng = np.random.RandomState(42)
age_null_random_list = rng.uniform(age_avg - age_std, age_avg + age_std, size=age_null_count)
train['Age'][np.isnan(train['Age'])] = age_null_random_list
```

**7. binning with cut**
```python
train['Age'] = pd.cut(train['Age'], 5)
```

**8. apply regex**
using previously mentioned `get_title` function:
```python
train['Title'] = train['Name'].apply(get_title)
```

**9. replace value**
```python
train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                        'Rare')
```

**10. replace value**
```python
train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')
```

**11. replace value**
```python
train['Title'] = train['Title'].replace('Mme', 'Mrs')
```

**12. fill Nans with 0**
```python
train['Title'] = train['Title'].fillna(0)
```

**13. drop columns**
```python
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 
                 'SibSp', 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis=1)
```

**14. encode labels**
```python
from sklearn.preprocessing import LabelEncoder

# encode labels
le = LabelEncoder()

le.fit(train['Sex'])
train['Sex'] = le.transform(train['Sex'])

le.fit(train['Title'])
train['Title'] = le.transform(train['Title'])

le.fit(train['Embarked'].values)
train['Embarked'] = le.transform(train['Embarked'].values)

le.fit(train['Fare'])
train['Fare'] = le.transform(train['Fare'])

le.fit(train['Age'])
train['Age'] = le.transform(train['Age'])
```

Great! Now that we have defined all preprocessing operations, let's gather them under the `Dataloader` class:

`utils/dataloader.py`
```python
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    # apply regex
    def get_title(self, name):
        pattern = ' ([A-Za-z]+)\.'
        title_search = re.search(pattern, name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    def load_data(self):
        # columns combination
        self.dataset['FamilySize'] = self.dataset['SibSp'] + self.dataset['Parch'] + 1

        # replace value
        self.dataset['IsAlone'] = 0
        self.dataset.loc[self.dataset['FamilySize'] == 1, 'IsAlone'] = 1

        # fill Nan with mode
        self.dataset['Embarked'] = self.dataset['Embarked'].fillna(self.dataset['Embarked'].mode()[0])

        # fill Nan with median
        self.dataset['Fare'] = self.dataset['Fare'].fillna(self.dataset['Fare'].median())
        # binning with qcut
        self.dataset['Fare'] = pd.qcut(self.dataset['Fare'], 4)

        # fill Nan with values from random distribution
        age_avg = self.dataset['Age'].mean()
        age_std = self.dataset['Age'].std()
        age_null_count = self.dataset['Age'].isnull().sum()
        rng = np.random.RandomState(42)
        age_null_random_list = rng.uniform(age_avg - age_std, age_avg + age_std, size=age_null_count)
        self.dataset['Age'][np.isnan(self.dataset['Age'])] = age_null_random_list

        # binning with cut
        self.dataset['Age'] = pd.cut(self.dataset['Age'], 5)

        # apply regex
        self.dataset['Title'] = self.dataset['Name'].apply(self.get_title)
        # replace
        self.dataset['Title'] = self.dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                               'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                               'Rare')
        # replace
        self.dataset['Title'] = self.dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        # replace
        self.dataset['Title'] = self.dataset['Title'].replace('Mme', 'Mrs')
        # fill nans
        self.dataset['Title'] = self.dataset['Title'].fillna(0)

        # drop columns
        drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',
                         'Parch', 'FamilySize']

        self.dataset = self.dataset.drop(drop_elements, axis=1)

        # encode labels
        le = LabelEncoder()

        le.fit(self.dataset['Sex'])
        self.dataset['Sex'] = le.transform(self.dataset['Sex'])
        
        le.fit(self.dataset['Title'])
        self.dataset['Title'] = le.transform(self.dataset['Title'])

        le.fit(self.dataset['Embarked'].values)
        self.dataset['Embarked'] = le.transform(self.dataset['Embarked'].values)

        le.fit(self.dataset['Fare'])
        self.dataset['Fare'] = le.transform(self.dataset['Fare'])

        le.fit(self.dataset['Age'])
        self.dataset['Age'] = le.transform(self.dataset['Age'])

        return self.dataset
```
That's how the data looks like after preprocessing:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/preprocessed_train.jpg?raw=true">
</div>

Let's consider the parts which are needed to check your project. 

### Specifications

[**Here**](Specifications_DOC.md) you will find a complete guide to filling `specifications.json`
Read it carefully and follow all given instructions

For automatic project checking, we have set some limitations to the data preprocessing module and defined a bunch of operations allowed to perform on data. To check the correction of your data preprocessing operations we created our own correct implementation of all necessary operation. So at each data preprocessing step we run your implementation and proper one and then we compare the results. 

That is why after completing the dataloader, you'll need to create `specifications.json` where you'll describe all operations you've performed on the data, so we will be able to reproduce them with our implementation.

> **Important note:**
Note, that you can pass a list of columns names and other parameters to each operation, so think about the correct order of operations to opimize their quantity

For our DataLoader is the final look of the `settings/specifications.json`:
```
{
    "description":{
        "X": [
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked"
        ],
        "final_columns":[
                    "Sex",
                    "Title",
                    "Embarked",
                    "Fare",
                    "Age"
                ],
        "y":"Survived",
        "metrics":"accuracy_score"
    },
    "operations":[
        {
            "operation_number":1,
            "operation_name":"columns_combination",
            "params":{
                "in_columns_list":[
                    [
                        "SibSp",
                        "Parch"
                    ]
                ],
                "out_columns":[
                    "FamilySize"
                ],
                "coefficients_list":[
                    [
                        1,
                        1
                    ]
                ],
                "biases":[
                    1
                ],
                "methods":[
                    "addition"
                ]
            }
        },
        {
            "operation_number":2,
            "operation_name":"replace",
            "params":{
                "in_columns":[
                    "IsAlone"
                ],
                "old_values":[
                    0
                ],
                "new_values":[
                    1
                ],
                "condition_columns":[
                    "FamilySize"
                ],
                "conditions":[
                    "equal"
                ],
                "condition_values":[
                    1
                ],
                "defaults":[
                    0
                ],
                "inplaces":[
                    "False"
                ]
            }
        },
        {
            "operation_number":3,
            "operation_name":"apply_regex",
            "params":{
                "in_columns":[
                    "Name"
                ],
                "out_columns":[
                    "Title"
                ],
                "methods":[
                    "search"
                ],
                "patterns":[
                    " ([A-Za-z]+)\\."
                ],
                "groups":[
                    1
                ],
                "inplaces":[
                    "False"
                ]
            }
        },
        {
            "operation_number":4,
            "operation_name":"replace",
            "params":{
                "in_columns":[
                    "Title",
                    "Title",
                    "Title"
                ],
                "old_values":[
                    [
                        "Lady",
                        "Countess",
                        "Capt",
                        "Col",
                        "Don",
                        "Dr",
                        "Major",
                        "Rev",
                        "Sir",
                        "Jonkheer",
                        "Dona"
                    ],
                    [
                        "Mlle",
                        "Ms"
                    ],
                    "Mme"
                ],
                "new_values":[
                    "Rare",
                    "Miss",
                    "Mrs"
                ],
                "inplaces":[
                    "True",
                    "True",
                    "True"
                ]
            }
        },
        {
            "operation_number":5,
            "operation_name":"fill_nans",
            "params":{
                "in_columns":[
                    "Fare",
                    "Age",
                    "Title",
                    "Embarked"
                ],
                "methods":[
                    "qcut",
                    "random",
                    "zero",
                    "mode"
                ]
            }
        },
        {
            "operation_number":5,
            "operation_name":"bins",
            "params":{
                "in_columns":[
                    "Fare",
                    "Age"
                ],
                "methods":[
                    "qcut",
                    "cut"
                ],
                "bins_nums":[
                    4,
                    5
                ],
                "inplaces":[
                    "True",
                    "True"
                ]
            }
        },
        {
            "operation_number":6,
            "operation_name":"drop_columns",
            "params":{
                "in_columns":[
                    "PassengerId",
                    "Name",
                    "Ticket",
                    "Cabin",
                    "SibSp",
                    "Parch",
                    "FamilySize"
                ]
            }
        },
        {
            "operation_number":7,
            "operation_name":"encode_labels",
            "params":{
                "in_columns":[
                    "Sex",
                    "Title",
                    "Embarked",
                    "Fare",
                    "Age"
                ]
            }
        }
    ]
}
```
> **Important note:**
Use **Mean Absolute Percentage Error** for regression task. Put `mean_absolute_percentage_error` into the metrics specification. Implementation:
```python
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

Second is a `Dataset` class that contains methods to give us some information about your dataset and allows us to generate data samples.

`utils/dataset.py`
```python
import pandas as pd
import csv


class Dataset:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def len(self):
        """
        Get number of examples
        @return: int
        """
        with open(self.csv_file, 'rt') as f:
            return sum(1 for row in f) - 1

    def columns(self):
        """
        Get list of columns names
        @return: list
        """
        with open(self.csv_file, 'rt') as f:
            columns = f.readline().rstrip().split(',')
        del columns[1]
        return columns

    def getitem(self, index):
        """
        Get example by index
        @param index: int
        @return: list, int
        """
        'Generates one sample of data'
        # Select sample
        idx = index + 1
        # Load data and get label
        with open(self.csv_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                if str(idx) in line:
                    break

        y = int(line[1])
        del line[1]
        x = line
        return x, y

    def get_items(self, items_number):
        """
        Get specific amount of examples
        @param items_number:
        @return: pd.DataFrame, pd.Series
        """
        data = pd.read_csv(self.csv_file, nrows=items_number)
        y = data['Survived']
        x = data.drop(['Survived'], axis=1)
        return x, y
```

## Model Selection, Training and Saving

Firstly, let's load our data:
```python
from utils.dataloader import DataLoader

X_raw = train.drop("Survived", axis=1)

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
y = train["Survived"]
```

To solve our task, we need to choose an estimator that will classify the data. Let's take a bunch of estimators, train them and compare the results (here we used **accuracy score**):

```python
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns = log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x = 'Accuracy', y = 'Classifier', data = log, color = "b")
log
```
Here are the result of each model:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/accurancy_log.jpg?raw=true">
</div>

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/figures/fare.png?raw=true">
</div>

We can see that `SVC` shows the best results: accuracy score is `0.82` on the validation set.

Now we need to train the estimator on the whole training set and save the model:
```python
import pickle
import json
import pandas as pd
from sklearn.svm import SVC

from utils.dataloader import DataLoader 
from settings.constants import TRAIN_CSV


with open('settings/specifications.json') as f:
    specifications = json.load(f)

raw_train = pd.read_csv(TRAIN_CSV)
x_columns = specifications['description']['X']
y_column = specifications['description']['y']

X_raw = raw_train[x_columns]

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
y = raw_train.Survived

model = SVC()
model.fit(X, y)
with open('models/SVC.pickle', 'wb')as f:
    pickle.dump(model, f)
```

The trained model is saved now. To check its performance we can use the following code:
```python
import pickle
import json
import pandas as pd
from sklearn.svm import SVC

from utils.dataloader import DataLoader 
from settings. constants import VAL_CSV


with open('settings/specifications.json') as f:
    specifications = json.load(f)

x_columns = specifications['description']['X']
y_column = specifications['description']['y']

raw_val = pd.read_csv(VAL_CSV)
x_raw = raw_val[x_columns]

loader = DataLoader()
loader.fit(x_raw)
X = loader.load_data()
y = raw_val.Survived

loaded_model = pickle.load(open('models/SVC.pickle', 'rb'))
loaded_model.score(X, y)
```

The `accuracy_score` is `~0.827...`

Next, you'll need to create two simple scripts for training the model and for using the saved model.
`utils/trainer.py`:
```python
from sklearn.svm import SVC


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return SVC().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
```

`utils/predictor.py`:
```python
import pickle

from settings.constants import SAVED_ESTIMATOR


class Predictor:
    def __init__(self):
        self.loaded_estimator = pickle.load(open(SAVED_ESTIMATOR, 'rb'))

    def predict(self, data):
        return self.loaded_estimator.predict(data)
```
Don't forget to create `__init__.py` for the `utils` package.
`utils/__init__.py`:
```python
from .dataloader import DataLoader
from .dataset import Dataset
from .trainer import Estimator
from .predictor import Predictor
```
Now all is left is an API.

## API

Our API will have only one route `/predict`, which will receive some validation data and return prediction. The body of the request contains `data` field which stores json with the data. Here is the full code:
```python
from utils import Predictor
from utils import DataLoader

from flask import Flask, request, jsonify, make_response

import pandas as pd
import json


app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    received_keys = sorted(list(request.form.keys()))
    if len(received_keys) > 1 or 'data' not in received_keys:
        err = 'Wrong request keys'
        return make_response(jsonify(error=err), 400)

    data = json.loads(request.form.get(received_keys[0]))
    df = pd.DataFrame.from_dict(data)

    loader = DataLoader()
    loader.fit(df)
    processed_df = loader.load_data()

    predictor = Predictor()
    response_dict = {'prediction': predictor.predict(processed_df).tolist()}

    return make_response(jsonify(response_dict), 200)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000) 
```
Almost there. Let's give it a final check:
```python
import json
import requests
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import DataLoader, Estimator 
from settings. constants import TRAIN_CSV, VAL_CSV

with open('settings/specifications.json') as f:
    specifications = json.load(f)
    
info = specifications['description']
x_columns, y_column, metrics = info['X'], info['y'], info['metrics']

train_set = pd.read_csv(TRAIN_CSV, header=0)
val_set = pd.read_csv(VAL_CSV, header=0)

train_x, train_y = train_set[x_columns], train_set[y_column]
val_x, val_y = val_set[x_columns], val_set[y_column]

loader = DataLoader()
loader.fit(val_x)
val_processed = loader.load_data()
print('data: ', val_processed[:10])

req_data = {'data': json.dumps(val_x.to_dict())}
response = requests.get('http://0.0.0.0:8000/predict', data=req_data)
api_predict = response.json()['prediction']
print('predict: ', api_predict[:10])

api_score = eval(metrics)(val_y, api_predict)
print('accuracy: ', api_score)
```
Here is the output:
```
    data:  [[1 0 1 3 0 0 3]
            [1 1 2 2 2 1 2]
            [3 0 1 2 0 0 1]
            [3 1 1 0 1 1 2]
            [3 1 1 0 2 0 2]
            [2 1 2 2 2 1 2]
            [2 0 3 2 2 1 3]
            [1 1 2 2 2 1 2]
            [1 0 1 3 2 1 1]
            [3 1 1 2 0 0 0]]
    predict:  [1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
    accuracy:  0.8272251308900523
```
Great! Let's create `requirements.txt` and Dockerize the app:

`requirements.txt`:
```
Flask
numpy
pandas
scikit_learn
```
`Dockerfile`:
```
FROM python:3.7

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN export PYTHONPATH='${PYTHONPATH}:/app'

COPY . .

CMD ["python", "./app.py"]
```
Build the image using:

`docker build -t <user-name>/<name-of-the-container>:<tag-name> .`

Then push it to the [Docker Hub](https://hub.docker.com/):

`docker push <user-name>/<name-of-the-container>:<tag-name>`

And submit your project with `<user-name>/<name-of-the-container>:<tag-name>`
Now it's time to come up with a task and data, and implement all the steps from the tutorial for your own project. Good luck, have fun!       

If you have any questions, write `@DRU Team` in Slack!
