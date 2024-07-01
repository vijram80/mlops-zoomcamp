import batch
from datetime import datetime
import pandas as pd
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
categorical = ['PULocationID', 'DOLocationID']
df_input = pd.DataFrame(data, columns=columns)

#batch.save_data('2023-01', df_input)

df_output = batch.read_data('2023-01', categorical)
batch.save_data('results', df_input)
print(df_output['duration'].sum())

