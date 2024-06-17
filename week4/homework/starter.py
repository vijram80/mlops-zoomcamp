#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def predict(df: pd.DataFrame):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

def std_deviation(y_pred):
    return y_pred.std()

def mean(y_pred):
    return y_pred.mean()



#df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    input_file=f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)
    predicted_value = predict(df)
    print(mean(predicted_value))

if __name__ == '__main__':
    run()



