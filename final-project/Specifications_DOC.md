# Specification.json guide
*Here is the most responsible part of your Final Project. In order to validate your code correctly, we're asking you to describe your data and `DataLoader()` operations in special file called **specification.json*** 
*Reading this specification, Bot will try to apply it's own reference functions to your data and compare the result with yours*
*So, in this guide we will clearly explain how to fill the specification to pass your Final Project and **get your long-awaited certificate!***

## Table Of Contents
- [File structure](#file-structure)
- ["Params"](#params)
	- [Example](#example)
	- [`"handle_outliers"`](#handle_outliers)
	- [`"fill_nans"`](#fill_nans)
	- [`"bins"`](#bins)
	- [`"replace"`](#replace)
	- [`"columns_combination"`](#columns_combination)
	- [`"process_dates"`](#process_dates)
	- [`"process_text"`](#process_text)
	- [`"apply_regex"`](#apply_regex)
	- [`"log_transform"`](#log_transform)
	- [`"normalize"`](#normalize)
	- [`"standardize"`](#standardize)
	- [`"drop_columns"`](#drop_columns)
	- [`"select_columns"`](#select_columns)
	- [`"encode_labels"`](#encode_labels)
- [Cheat sheet](#cheat-sheet)

## File structure

>**If you are not familiar with .json files - don't worry**
You can considered it as a big nested Python dictionary (just for understanding) 

File have 2 main keys: 
- `"description"` - contains main information about your data
- `"operations"` - contains information about data operations in your DataLoader
---
`"description"` contain 4 it's own keys:
- `"X"` - contains list of your initial data columns names
- `"final_columns"` - contains list of data columns names after DataLoader preprocessing 
- `"y"`  - contains name of column with target values
- `"metrics"` - obviously, contains name of metric you trying to maximize 

Available metrics: `"mean_absolute_percentage_error"` or `"accuracy_score"`

`"operations"` should contain list of your data operations
>**Important Note:**
>If your specification contain less then 4 **unique** operations, Bot will reject it!
>**Example:** 
If you have only `"drop_columns"`, `"fill_nans"` and `"encode_labels"` - Bot will reject it
>But if you have `"drop_columns"`, `"fill_nans"`, `"encode_labels"` and `"bins"` - Bot will accept it

Every operation should have these 3 keys:
- `"operation_number"` -  index number of operation 
- `"operation_name"` - name of current operation 
- `"params"` - contains parameters of current operation

Let's take a closer look to `"params"` 

## Params
`"params"` are individual for each operation and now we will be described in full

**Here you could see full list of available operation names with corresponding parameters:**

```json
{
    "handle_outliers":[
        "in_columns",
        "modes",
        "methods",
        "factors",
        "upper_quantiles",
        "lower_quantiles"
    ],
    "fill_nans":[
        "in_columns",
        "methods",
        "custom_values"
    ],
    "bins":[
        "in_columns",
        "bins_nums",
        "methods",
        "conditions",
        "choices",
        "defaults",
        "inplaces"
    ],
    "replace":[
        "in_columns",
        "old_values",
        "new_values",
        "condition_columns",
        "conditions",
        "condition_values",
        "defaults",
        "inplaces"
    ],
    "columns_combination":[
        "in_columns_list",
        "out_columns",
        "methods",
        "coefficients_list",
        "biases"
    ],
    "process_dates":[
        "in_columns",
        "date_formats",
        "timestamps"
    ],
    "apply_regex":[
        "in_columns",
        "out_columns",
        "methods",
        "patterns",
        "inplaces",
        "maxsplits",
        "rep_values",
        "counts",
        "indexes",
        "groups"
    ],
    "log_transform":[
        "in_columns"
    ],
    "normalize":[
        "in_columns"
    ],
    "standardize":[
        "in_columns"
    ],
     "drop_columns":[
        "in_columns"
    ],
    "select_columns":[
        "in_columns"
    ],
    "encode_labels":[
         "in_columns"
     ]
}
```
>**Important Note:**
>Unfortunately, you could use only these set of operations or their combinations
>Otherwise, if you use other operation and/or don't noticed operation in specifications - Bot will reject your project :(
>However, you can be sure, this set of operations will be sufficient for most datasets

But before we start describing each operation, let us give you an example of **specification.json** for "Titanic" dataset from [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)
Just using operations given in this example (with corresponding parameters) maybe enough for many datasets

### Example:
```json
{
    "description":{
        "X": [
            "PassengerId",
            "Pclass",
            "Name",
            "Gender",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked"
        ],
        "final_columns":[
                    "Gender",
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
                    "Gender",
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
Fine, let's look closer to the each operation
>**Hint:** 
>You may navigate via [Table of contents](#table-of-contents)

---
### `"handle_outliers"`
Applying simple outliers handler

**Example:**
Create Box plot for `'Age'` column in "Titanic" dataset:


<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_1_handle_out.jpg?raw=true">
</div>


And after removing outliers your "Age" Box plot should looks like:


<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_2_handle_out.jpg?raw=true">
</div>

In order to remove outliers, you have to define upper and lower limits, after/before which you treat the point as outlier. Here are 2 ways:
1. `"modes":["std"]` 
	Using formula: 
	`upper_limit = mean + std * factor`
	`lower_limit = mean - std * factor`
	
2. `"modes":["percentile"]` 
	Using specific quantiles:
	`upper_limit = df[column].quantile(upper_quantile)`
	`lower_limit = df[column].quantile(lower_quantile)`

And also you have to decide the method, using which you will remove outliers:
1. `"methods":["drop"]` 
	Just drop rows, that contain outliers 
	
2. `"methods":["cap"]` 
	Replace outliers with upper or lower limits 

Operation contains keys:
- `"in_columns"` - list of columns, to which you applying outliers handler
- `"modes"` - `"std"` or `"percentile"`  - way to find upper and lower limits
- `"methods"` - `"drop"` or `"cap"` - way to remove outliers
- `"factors"` - float used only in `"modes":["std"]` (check formula)
- `"upper_quantiles"` - float in [0; 1] used only in `"modes":["percentile"]` (check formula)
- `"lower_quantiles"` - float in [0; 1] used only in `"modes":["percentile"]` (check formula)

**Example:** 
```json
{
 "operation_number":1,
 "operation_name":"handle_outliers",
 "params":{
     "in_columns":[
         "Age"
     ],
     "modes":[
         "std"
     ],
     "methods":[
          "cap"  
     ],
     "factors":[
         1.5
     ]
 }
}
```
---
### `"fill_nans"`
Apllying simple NaN filling function
>Example of usage you could find in [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)

Operation contains keys:
- `"in_columns"` - list of columns, to which you are applying NaNs filling
- `"methods"` - method to fill NaNs
- `"custom_values"` - custom values to put instead NaNs only for `"methods":"custom"`

You have to define the method of NaNs filling. You could choose from:
1. `"methods":"zero"` - replacing all NaNs in column with 0
2. `"methods":"mean"` - replacing all NaNs in column with it's mean
3. `"methods":"mode"` - replacing all NaNs in column with it's mode
4. `"methods":"median"` - replacing all NaNs in column with it's median
5. `"methods":"custom"` - replacing all NaNs in column with some custom value, that should be put into parameter `"custom_values"`
6. `"methods":"random"` - replacing all NaNs in column with some random values
In `"methods":"random"` for numerical values you should use:
```python
rng = np.random.RandomState(42)
null_random_list = rng.uniform(avg - std, avg + std, size=null_count)
```
And for string values take random values from this column using:
```python
col_vals = list(dataset[~dataset[column].isnull()][column].unique())
null_random_list = [random.choice(col_vals) for i in range(null_count)]
```

**Example:**
```json
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
}
```
---
### `"bins"`
Applying split column values on bins 
>Example of usage you could find in [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)

Operation contains keys:
- `"in_columns"` - list of columns, to which you are applying split on bins
- `"bins_nums"` - number of bins 
- `"methods"` - you may choose between `"cut"` and `"qcut"` - that correspond to Pandas `pd.cut` and  `pd.qcut` with no additional parameters except data and bins number
- `"inplaces"` - `True` is for apply changes to this column, `False` is for create new column with name `<in_column_name> + "_categorical"`

**Example:**
```json
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
}
```
---
### `"replace"`
Replacing values in column depending on values in other column

>Example of usage you could find in [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)

Firstly you should decide, will you just replace `"old_values"` with `"new_values"` or you will use some `"conditions"` for replacing
 
Operation contains keys:
- `"in_columns"` - list of columns, to which you are applying replace
- `"old_values"` - old values, you want to be replaced. Only for non-conditional replacement
- `"new_values"` - values, you want to replace with
- `"condition_columns"` - column name. Only for conditional replacement
- `"conditions"` - condition under which the replacement occurs. Only for conditional replacement
- `"condition_values"` - condition value, that will be used for evaluate `"conditions"`. Only for conditional replacement
- `"defaults"` - default values, only for `"inplaces":"False"`. Only for conditional replacement
- `"inplaces"` - `True` is for replace values in **existing** column (`"in_column"`), `False` is for **create new column** with name `"in_column"` and default values `"defaults"` 

`"conditions"` can be:
- `equal` - Is for `==`
- `greater` - Is for `>`
- `lower` - Is for `<`
- `gte` - Is for `>=`
- `lte` - Is for `<=`


For clarification, code:
```python
dataset.loc[dataset[condition_column] == condition_value, in_column] = new_value
```
Should be described as:
```json
{
    "operation_number":1,
    "operation_name":"replace",
    "params":{
        "in_columns":[
            "column_name"
        ],
        "new_values":[
            "new_value"
        ],
        "condition_columns": [
	        "condition_column"
        ],
        "conditions": [
	        "equal"
        ],
        "condition_values":[
	        "condition_value"
        ],                
        "inplaces":[
            "True"
        ]
    }
}
```



**Example:**
```json
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
}
```
---
### `"columns_combination"`
Combine values in given 2 columns in given way (add, substract, multiplicate, divide)

>Example of usage you could find in [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)
 
Operation contains keys:
- `"in_columns_list"` - list of 2 columns, you want to combine
- `"out_columns"` - name of output column
- `"methods"` - can be only `"addition"`, `"substraction"` (that means `in_columns[0] - in_columns[1]`), `"multiplication"`, `"division"` 
- `"coefficients_list"` - list of 2 values, that will be coefficients for column values
- `"biases"` - bias value

**Example:**
```json
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
}
```
---
### `"process_dates"`
Splitting date column `"in_columns"` into day, weekday, month and year columns

Operation contains keys:
- `"in_columns"` - list of date columns, you want to split
- `"date_formats"` - RegEx format of date values in column (usually '%d.%m.%Y') 

New columns should be named:
- `"in_column" + "_year"` - which correspond to `pd.dt.year`
- `"in_column" + "_month"` - which correspond to `pd.dt.month`
- `"in_column" + "_weekday"` - which correspond to `pd.dt.weekday`
- `"in_column" + "_day"` - which correspond to `pd.dt.dayofyear`

And `"in_column"` should be replaced using:
`dataset[in_column] = pd.to_datetime(dataset[in_column], format=date_format)`

**Example:**
`"in_column"` looks like:


<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_3_date.jpg?raw=true">
</div>

After processing "in_column" is:


<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_4_date.jpg?raw=true">
</div>

And new columns appear:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_5_date.jpg?raw=true">
</div>

```json
{
    "operation_number":1,
    "operation_name":"process_dates",
    "params":{
        "in_columns_list":[
            "date"
        ],
        "date_formats":[
            "%x"
        ]
    }
}
```
---
### `"apply_regex"`
Applying regex to a column

>Example of usage you could find in [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)
 
Operation contains keys:
- `"in_columns"` - list of columns names, to which you will apply RegEx
- `"out_columns"` - list of output columns names
- `"methods"` - RegEx method (`"search"`, `"split"` or `sub`)
- `"patterns"` - RegEx pattern
- `"inplaces"` - `True` is for replacing `"in_columns"` with new values. `"False"` is for put new values in new columns with name `"out_columns"`
- `"maxsplits"` - argument for `"methods": "split"` (for `re.split`)
- `"rep_values"`- replacement value for `"methods": "sub"` (for `re.sub`)
- `"counts"` - `count` argument for `"methods": "sub"` (for `re.sub`)
- `"group"` - argument in `re.group()` method (check example)
- `"index"` - return only `str[index]` in the end

**Example:**

```
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
}
```



---
### `"log_transform"`
Applying logarithm transformation to column values using formula:
`new_column_value = np.log(in_column_value + 1)`

Operation key:
-`"in_columns"` - list of column, to which you want to apply `"log_transform"`

**Example:**
Titanic dataset, column "Fare":


<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_6_log.jpg?raw=true">
</div>

After applying `"log_transform"`:

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_7_log.jpg?raw=true">
</div>

```json
{
    "operation_number":1,
    "operation_name":"log_transform",
    "params":{
        "in_columns_list":[
            "Fare"
        ]
    }
}
```
---
### `"normalize"`
Applying normalization (or min-max normalization), that means scale all values in a fixed range between 0 and 1, to column using formula:
`new_column_value = (in_column_value - in_column.min()) / (in_column.max() - in_column.min())`

Operation key:
-`"in_columns"` - list of column, to which you want to apply `"normalize"`

**Example:**
Titanic dataset, column "Fare":

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_6_log.jpg?raw=true">
</div>

After applying `"normalize"`:


<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_8_norm.jpg?raw=true">
</div>

```json
{
    "operation_number":1,
    "operation_name":"normalize",
    "params":{
        "in_columns_list":[
            "Fare"
        ]
    }
}
```
---
### `"standardize"`
Applying z-code standardization, that means scale all values in a fixed range between -1 and 1, to column values using formula:
`new_column_value = (in_column_value - in_column.mean()) / in_column.std()`

Operation key:
-`"in_columns"` - list of column, to which you want to apply `"standardize"`

**Example:**
Titanic dataset, column "Fare":

<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_6_log.jpg?raw=true">
</div>

After applying `"standardize"`:


<div align="center">
    <img align="center" src="https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/spec_img/IMG_9_stand.jpg?raw=true">
</div>

```json
{
    "operation_number":1,
    "operation_name":"standardize",
    "params":{
        "in_columns_list":[
            "Fare"
        ]
    }
}
```
---
### `"drop_columns"`
Dropping all specified columns

>Example of usage you could find in [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)

Operation keys:
- `in_columns` - list of columns, you want to drop

**Example:**
```json
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
}
```
---
### `"select_columns"`
Dropping all not specified columns

Operation keys:
- `in_columns` - list of columns, you want to keep

**Example:**
Given operation doing same processing as `drop_columns` in last example
```json
{
    "operation_number":1,
    "operation_name":"select_columns",
    "params":{
        "in_columns":[
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "Fare",
            "Embarked",
            "IsAlone",
            "Title"
        ]
    }
}
```
---
### `"encode_labels"`
Encode labels using `sklearn.preprocessing.LabelEncoder` with replacing old values

>Example of usage you could find in [Final Project guide](https://github.com/DataRootUniversity/ds-fundamentals/blob/master/final-project/DOC.md)

Operation keys:
- `in_columns` - list of columns, you want to encode

**Example:**
```json
{
    "operation_number":7,
    "operation_name":"encode_labels",
    "params":{
        "in_columns":[
            "Gender",
            "Title",
            "Embarked",
            "Fare",
            "Age"
        ]
    }
}
```


## Cheat sheet
Here is all description are concentrated in one place:
```
{ 
    // handling outliers in different ways
    "handle_outliers":[
        "in_columns", // input column name(type: list[str]) 
        "modes",  // mode of handling outliers: `drop`, `cap` (type: list[str])
        "methods", // methods of handling outliers: `std`, `percentile` (type: list[str])
        "factors", // factor value in case using `std` (type: list[float])
        "upper_quantiles", // in case using `percentile` (type: list[float])
        "lower_quantiles"  // in case using `percentile` (type: list[float])
    ],
        // filling Nans in different ways
    "fill_nans":[
        "in_columns", // input column names (type: list[str]) 
        "methods", // methods of filling Nans (type: list[str]): 
                                        // `zero` - fill Nans with 0
                                        // `mean` - fill Nans with column mean
                                        // `mode` - fill Nans with column mode
                                        // `median` - fill Nans with column median
                                        // `custom` - fill Nans with custom value
                                        // `random` - fill Nans with random values in range (avg - std, avg + std) for numeric column type, random column values for string column type
        "custom_values" // custom values for filling Nans (type: dict[str:int/float]) 
    ],
        // split column values on bins inplace or by adding new column with name 'in_column_categorical'
    "bins":[
        "in_columns", // input column name (type: list[str]) 
        "bins_nums", // number of bins (type: list(int))
        "methods", // method of splitting (type: list[str]): `cut`, `qcut`, `condition` 
        "conditions", // conditions for case of using `condition` method (type: list[str])
        "choices", // choices for case of using `condition` method (type: list[str])
        "defaults", // default value (type: list[any type])
        "inplaces" // change `in_columns[i]` of add new column `in_column[i]_categorical` (type: list(bool))
    ],
        // replace values in column depending on values in other column
    "replace":[
        "in_columns", // input column name (dependent column)(type: list[str]) 
        "old_values", // value to replace (type: list[list/int/float/str])
        "new_values", // new value (type: list[int/float/str])
        "condition_columns", // condition column (independent column)(type: list[str]) 
        "conditions", // condition (type: list[str]): `equal`, `greater`, `lower`, `gte`, `lte`
        "condition_values", // condition value (type: list[int/float])
        "defaults", // default value (type: list[int/float/str])
        "inplaces" // (type: list[bool]) if inplace: replace `in_column` values, else: create `in_column` with `default` value   
    ],
        // combine columns in different ways
    "columns_combination":[
        "in_columns_list", // input columns names (type: list[list, len = 2])
        "out_columns", // output column name (type: list[str]) 
        "methods", // combination method for numeric values (type: list[str]): 
                                        // `addition` - add columns values
                                        // `subtraction` - substract columns values (`in_columns[0]` - `in_columns[1]`)
                                        // `multiplication` - multiply columns values
                                        // `division` - divide `in_columns[0]` values on `in_columns[1]` values
        "coefficients_list", // columns values coefficients (type: list[list[int/float], len = 2]))
        "biases" // bias value (type: list[int/float])
    ],
        // dates processing: transforms date column into `column_name_year`, 
                                                      // `column_name_month`, 
                                                      // `column_name_weekday`, 
                                                      // `column_name_doy` 
        // if the column is `timestamp` it'll also return `column_name_time`,
                                                       // `column_name_doy_sin`,
                                                       // `column_name_doy_cos`
    "process_dates":[
        "in_columns", // input column name (type: list[str]) 
        "date_formats" // format of date values in column (type: list[str]), e.g. '%d.%m.%Y' is for '01.01.2020' 
    ],
        // applying regex to string values 
    "apply_regex":[
        "in_columns", // input column name (type: list[str]) 
        "out_columns", // output column name (type: list[str])
        "methods", // regex method (type: list[str]):
                                 // `search`
                                 // `split`
                                 // `sub`
        "patterns", // regex pattern (type: list[str])
        "inplaces", // change `in_column` or create `out_column` (type: list(bool)) 
        "maxsplits", // `maxsplit` arg for `re.split` (type: list(int))
        "rep_values", // replacement value for `re.sub` (type: list(str))
        "counts", // `count` argument `re.sub` (type: list(int))
        "indexes", // (type: list)
        "groups" // (type: list[int])
    ],
        // logarithm transformation
    "log_transform":[
        "in_columns" // input columns names (type: list of strings)
    ],
        // normalization (or min-max normalization) scale all values in a fixed range between 0 and 1
    "normalize":[
        "in_columns" // input columns names (type: list of strings)
    ],
        // z-score standardization
    "standardize":[
        "in_columns" // input columns names (type: list of strings)
    ],
        // drop specific columns 
     "drop_columns":[
        "in_columns" // input columns names (type: list of strings)
    ],
        // drop all columns except specified
    "select_columns":[
        "in_columns" // input columns names (type: list of strings)
    ],
        // apply label encoder to specified columns
    "encode_labels":[
         "in_columns" // input columns names (type: list of strings)
     ]
}
```

We hope it was useful for you and you will pass Final Project soon
If you still have questions, contact us `@DRU Team` or ask your question in the `#8-final-project` chat
Good luck!
