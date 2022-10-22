# purchase-predictions
Predicting purchases from clickstream data and wrapped in Kedro package

## Overview
Can we build an accurate machine learning model to predict purchases using clickstream data?

## Analysis
- __Please see ./purchase-predictions/model/notebooks/main.ipynb__
- Steps to interact with notebook:
  - `git clone https://github.com/jacobweiss2305/purchase-predictions.git`
  - `cd purchase-predictions`
  - `python -m venv venv` 
    - (see [provision](./model/docs/provision.md) for virtualenv installation)
  - `pip install -r requirements.txt`
  - `source venv/bin/activate`
  - __To spin up Jupyter lab you must use kedro__:
       ```
       cd ./purchase-predictions/model/

       kedro jupyter lab
       ```
## Provision
- I recommend using pyenv for python version control. We are running 3.8.9. Please see [provision](./model/docs/provision.md) for details (model/docs/provision.md).
