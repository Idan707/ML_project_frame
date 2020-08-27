import pandas as pd
import featuretools as ft

import config
from featuretools.variable_types import Numeric, DatetimeTimeIndex, Id

# Custom primitives for featuretools
# Rolling Mean Windows
class RollingMean_3day(ft.primitives.TransformPrimitive):
    name = 'rolling_mean_3d'
    input_types = [Numeric, DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, numeric, time, id, win='3d'):
        super(RollingMean_3day, self).__init__(numeric, time, id)
        self.win = win

    def get_function(self):
        def rolling_mean_3d(numeric, time, id):
            data = {'numeric': numeric, 'id': id, 'time': time}
            df = pd.DataFrame(data).set_index('time')
            apply = lambda x: x.rolling(self.win, 1).mean()
            values = df.groupby('id')['numeric'].transform(apply)
            return values
        return rolling_mean_3d

class RollingMean_7day(ft.primitives.TransformPrimitive):
    name = 'rolling_mean_7d'
    input_types = [Numeric, DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, numeric, time, id, win='7d'):
        super(RollingMean_7day, self).__init__(numeric, time, id)
        self.win = win

    def get_function(self):
        def rolling_mean_7d(numeric, time, id):
            data = {'numeric': numeric, 'id': id, 'time': time}
            df = pd.DataFrame(data).set_index('time')
            apply = lambda x: x.rolling(self.win, 1).mean()
            values = df.groupby('id')['numeric'].transform(apply)
            return values
        return rolling_mean_7d

class RollingMean_21day(ft.primitives.TransformPrimitive):
    name = 'rolling_mean_21d'
    input_types = [Numeric, DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, numeric, time, id, win='21d'):
        super(RollingMean_21day, self).__init__(numeric, time, id)
        self.win = win

    def get_function(self):
        def rolling_mean_21d(numeric, time, id):
            data = {'numeric': numeric, 'id': id, 'time': time}
            df = pd.DataFrame(data).set_index('time')
            apply = lambda x: x.rolling(self.win, 1).mean()
            values = df.groupby('id')['numeric'].transform(apply)
            return values
        return rolling_mean_21d

# Rolling STD Windows
class RollingSTD_3day(ft.primitives.TransformPrimitive):
    name = 'rolling_STD_3d'
    input_types = [Numeric, DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, numeric, time, id, win='3d'):
        super(RollingSTD_3day, self).__init__(numeric, time, id)
        self.win = win

    def get_function(self):
        def rolling_STD_3d(numeric, time, id):
            data = {'numeric': numeric, 'id': id, 'time': time}
            df = pd.DataFrame(data).set_index('time')
            apply = lambda x: x.rolling(self.win, 1).std()
            values = df.groupby('id')['numeric'].transform(apply)
            return values
        return rolling_STD_3d

class RollingSTD_7day(ft.primitives.TransformPrimitive):
    name = 'rolling_STD_7d'
    input_types = [Numeric, DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, numeric, time, id, win='7d'):
        super(RollingSTD_7day, self).__init__(numeric, time, id)
        self.win = win

    def get_function(self):
        def rolling_STD_7d(numeric, time, id):
            data = {'numeric': numeric, 'id': id, 'time': time}
            df = pd.DataFrame(data).set_index('time')
            apply = lambda x: x.rolling(self.win, 1).std()
            values = df.groupby('id')['numeric'].transform(apply)
            return values
        return rolling_STD_7d

class RollingSTD_21day(ft.primitives.TransformPrimitive):
    name = 'rolling_STD_21d'
    input_types = [Numeric, DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, numeric, time, id, win='21d'):
        super(RollingSTD_21day, self).__init__(numeric, time, id)
        self.win = win

    def get_function(self):
        def rolling_STD_21d(numeric, time, id):
            data = {'numeric': numeric, 'id': id, 'time': time}
            df = pd.DataFrame(data).set_index('time')
            apply = lambda x: x.rolling(self.win, 1).std()
            values = df.groupby('id')['numeric'].transform(apply)
            return values
        return rolling_STD_21d

# run featuretools
def run():
    df = pd.read_csv(config.CRED_FILE, infer_datetime_format=True)
    df.rename(
        columns={
            'Transaction date': 'Transaction_date', 
            'isFradulent' : 'target'
                        
        }, inplace=True
                
    )

    df['Transaction_date'] = pd.to_datetime(df['Transaction_date'])
    df.sort_values('Transaction_date', inplace=True)

    es = ft.EntitySet(id='data')
    es = es.entity_from_dataframe(dataframe = df, 
                                  entity_id='trx_df', 
                                  time_index="Transaction_date",
                                  make_index = True,
                                  index = 'index',
                                  variable_types = {'Is declined' : ft.variable_types.Categorical,
                                                  'isForeignTransaction': ft.variable_types.Categorical,
                                                  'isHighRiskCountry': ft.variable_types.Categorical,
                                                  'Transaction_date': ft.variable_types.DatetimeTimeIndex,
                                                  'Merchant_id': ft.variable_types.Id},
                                  already_sorted=True)
    es.add_last_time_indexes()

    es.normalize_entity(base_entity_id='trx_df',
                        new_entity_id='Merchant_id',
                        index='Merchant_id')

    es.add_last_time_indexes()
    print(es.entity_dict.items())

    feature_matrix, feature_names = ft.dfs(entityset = es, 
                                       target_entity = 'trx_df',
                                       entities = 'trx_df',
                                       max_depth = 2,
                                       trans_primitives = [RollingSTD_21day,
                                                           'cum_mean','cum_max','month'], 
                                       agg_primitives = ['sum','count'])

    print('%d Total Features' % len(feature_names))
    #feature_matrix.to_csv('fm_tesing.csv')

if __name__ == "__main__":
    run()






