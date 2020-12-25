# Covid-19 data analysis and predict
Use simple model to analysis covid-19's behavier and predict Infected and deaths.

# Install
required packages in Python
```
pip install pandas,numpy ,xgboost,matplotlib.pyplot,scipy
```
download SIRD_funs.py and SIRD_DEMO.ipynb in your folder.

# Run
open the SIRD_DEMO.ipynb in jupyter notebook.
In this notebook's data analysis part, you just need to run each cell, then you can see the result.

In this notebook's prediction part,

First,given country which you want to use in training predict model and prediction day.

Next, select the country which you want to predict Infected and deaths.

Example:
```
from SIRD_funs import input_all_data
DATA=input_all_data()
from SIRD_funs import predict_train,predict_test
Country=['Netherlands','United Kingdom','France','Belgium','Spain','Italy','Germany','US','Egypt','Kenya','Japan','Austria','Qatar']
day=1
[bst_D,bst_I,reg_D,reg_I]=predict_train(DATA,Country,day)
predict_test(DATA,'Italy',bst_D,bst_I,reg_D,reg_I,day)
```
