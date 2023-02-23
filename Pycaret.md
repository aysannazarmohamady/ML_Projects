**First, about Pycaret:**
Compared with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with a few lines only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks, such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and a few more.

**Solving a classification problem with the PyCaret library:**

Install the Pycaret library
```python
!pip install pycaret
```

Read the dataset using the Pandas library:
```python
# importing pandas to read the CSV file
import pandas as pd
# read the data
data_classification = pd.read_csv('datasets/loan_train_data.csv')
# view the top rows of the data
data_classification.head()
```
![image](https://user-images.githubusercontent.com/30371881/220954032-8f79c96a-0091-4d57-8c6f-6981750fde3d.png)

Initial settings:
1. Import the module
2. Initial setup
```python
import numpy as np

# import the classification module
from pycaret import classification
# setup the environment
classification_setup = classification.setup(data= data_classification, target='Personal Loan')
```
![image](https://user-images.githubusercontent.com/30371881/220965908-b2c81bc4-a4ac-45a9-89d9-84cda6749c2f.png)


Machine learning model training with PyCaret library
```python
# build the decision tree model
classification_dt = classification.create_model('dt')
```
![image](https://user-images.githubusercontent.com/30371881/220966414-8eb61e3e-729c-429f-bac1-c0c4b91c1586.png)

Next, to train the XGBoost model, just need to add the "xgboost" string:
```python
# build the xgboost model
classification_xgb = classification.create_model('xgboost')
```








