if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy

@transformer
def transform(data, **kwargs) -> Tuple[DictVectorizer , LinearRegression, numpy.ndarray]:
    # Specify your transformation logic here
    
    categorical = ['PULocationID', 'DOLocationID']
    df_val = data
    val_dicts = df_val[categorical].to_dict(orient='records')
    
    dv = DictVectorizer()
    
    X_val = dv.fit_transform(val_dicts)

    target = 'duration'
    y_val = df_val[target].values
    
    lr = LinearRegression()
    lr.fit(X_val, y_val)
    
    y_pred = lr.predict(X_val)

    print(lr.intercept_)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return dv, lr, rmse


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'