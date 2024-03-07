import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import category_encoders as ce
from sklearn.model_selection import train_test_split

class FHVProcessPipeline():
    
    '''
    creates a dataset, and samples it
    '''
    def __init__(self, data_path, samples=100000, random_state=42):
        
        self.data=pd.read_parquet(data_path)
        self.data = self.data.sample(n=samples + int(samples * 0.25), random_state=random_state)
        self.original_data = self.data.copy()
        self.test_data = None
    
    def drop_columns(self, cols):
        self.data.drop(columns=cols, inplace=True)
        if self.test_data is not None:
            self.test_data.drop(columns=cols, inplace=True)

    
    def feature_engineering(self):
        self.data['is_airport_trip'] = np.where(self.data['airport_fee'] > 0, 1, 0)
        self.data['congestion_lvl'] = self.data['congestion_surcharge'].apply(self.calcular_congestion_surcharge)
        self.data['trip_time_real'] = round((self.data['dropoff_datetime'] - self.data['pickup_datetime']).dt.total_seconds() / 60.0,3) 
        self.data["total_fare"] = self.data["base_passenger_fare"] + self.data["congestion_surcharge"] + self.data["airport_fee"]
        self.data['pickup_datetime']=pd.to_datetime(self.data['pickup_datetime'])
        self.data['pickup_day_no']=self.data['pickup_datetime'].dt.weekday # monday 0 - sunday 6
        self.data['pickup_hour']=self.data['pickup_datetime'].dt.hour
        self.data['hourly_segments'] = self.data.pickup_hour.map({0:'H2',1:'H1',2:'H1',3:'H1',4:'H1',5:'H2',6:'H3',7:'H4',8:'H5',
                                     9:'H4',10:'H4',11:'H5',12:'H5',13:'H5',14:'H5',15:'H6',16:'H6',
                                     17:'H6',18:'H5',19:'H4',20:'H4',21:'H3',22:'H3',23:'H3'})

        self.data['day_segments'] = self.data.pickup_day_no.map({0:'WD',1:'WD',2:'WD',3:'WD',4:'WD',5:'WK',6:'WK'})
        self.data = pd.get_dummies(self.data, columns=['hourly_segments', 'day_segments'])
        bool_cols = [
                    'hourly_segments_H1', 'hourly_segments_H2', 'hourly_segments_H3',
                    'hourly_segments_H4', 'hourly_segments_H5', 'hourly_segments_H6',
                    'day_segments_WD', 'day_segments_WK']

        # https://www.kaggle.com/code/yasserh/uber-fare-prediction-comparing-best-ml-models
        for c in bool_cols:
            self.data[c] = self.data[c].astype(int)
            
    def drop_outliers(self):
        self.data = self.data[(self.data['base_passenger_fare'] > 5)]
        # Only consider trip time greater than 5 mins and less than 2 hours and trip mile greater than 1
        self.data = self.data[(self.data['trip_time_real'] > 5) & (self.data['trip_time_real'] < 120) & (self.data['trip_miles'] > 1)]  

    def split_data(self, test_size):
        X = self.data.drop('total_fare', axis=1)  # Assuming 'total_fare' is the target
        y = self.data['total_fare']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.data = pd.concat([X_train, y_train], axis=1)
        self.test_data = pd.concat([X_test, y_test], axis=1)
    
    def split_val_test(self):
        X = self.test_data.drop('total_fare', axis=1)  # Assuming 'total_fare' is the target
        y = self.test_data['total_fare']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        self.validate_data = pd.concat([X_train, y_train], axis=1)
        self.test_data = pd.concat([X_test, y_test], axis=1)
    
        
    def process_data(self, test_size=0.25):
        # step 1: drop unecessary columns
        cols_to_drop = ['hvfhs_license_num', 'dispatching_base_num', 'originating_base_num', 'shared_request_flag', 'shared_match_flag', 'access_a_ride_flag', 'wav_request_flag', 'wav_match_flag', 'bcf', 'tolls', 'sales_tax', 'tips', 'request_datetime', 'on_scene_datetime', 'driver_pay']
        
        self.drop_columns(cols_to_drop)
        
        self.original_data.drop(columns=cols_to_drop, inplace=True)
        
        # step 2: make new feature
        self.feature_engineering()
    
        # step 3: drop outliers
        self.drop_outliers()
        
        # step 4: feature cross
        self.ft_cross()
        
        # step 5: split data into training and test set
        self.split_data(test_size)
        
        # step 6: encode training and testset
        self.target_encode()
        self.target_encode_test()
        
        # clean up columns
        self.drop_columns(['day_x_time', 'PUxDOL'])
        
        # step 7: clear out columns
        self.drop_columns(['pickup_datetime', 'dropoff_datetime','PULocationID','DOLocationID',	'trip_time', 'base_passenger_fare', 'congestion_surcharge', 'airport_fee', 'pickup_day_no', 'pickup_hour'])
        
        # step 8: spit test set into validation and test set
        self.split_val_test()
        
        return self.data, self.validate_data, self.test_data, self.original_data
    
    def ft_cross(self):
        self.data['day_x_time'] = self.data['pickup_day_no'].astype(str) + self.data['pickup_hour'].astype(str).str.zfill(2)
        self.data['PUxDOL'] = self.data['PULocationID'].astype(str) + self.data['DOLocationID'].astype(str)
    
    def target_encode(self):
        self.dt_encoder = ce.TargetEncoder()
        self.data["fare_per_mile"] = (self.data["total_fare"] / self.data["trip_miles"]).round(2)
        self.data['enc_day_x_time'] = self.dt_encoder.fit_transform(self.data['day_x_time'], self.data['fare_per_mile'])
        self.data.drop(columns=['fare_per_mile'], inplace= True)
        self.loc_encoder = ce.TargetEncoder()
        self.data['enc_PUxDOL'] = self.loc_encoder.fit_transform( self.data['PUxDOL'],  self.data['total_fare'])

    def target_encode_test(self):
        self.test_data['enc_day_x_time'] = self.dt_encoder.transform(self.test_data['day_x_time'])
        self.test_data['enc_PUxDOL'] = self.loc_encoder.transform(self.test_data['PUxDOL'])
    
    def calcular_congestion_surcharge(self,congestion_surcharge):
        if congestion_surcharge == 0:
            return 0        # no congestion
        elif congestion_surcharge > 0 and congestion_surcharge < 2:
            return 1        # low
        elif congestion_surcharge >= 2 and congestion_surcharge < 3:
            return 2        # medium
        else:
            return 3        # high
    