import numpy as np
import pandas as pd

def mean_avg_precision_at_7(y_test, y_pred_proba):

    map7 = np.zeros(y_pred_proba.shape[0])
    idx = 0

    for row1, row2 in zip(y_test.values, y_pred_proba):
        top7 = np.argsort(row2)[::-1][:7]
        n_corr = 0
        precision = 0
        for idx2, item in enumerate(row1[top7]):
            if item == 1:
                n_corr +=1
                precision += n_corr/(idx2+1)
        if n_corr > 0:
            precision /= n_corr

        map7[idx] = precision
        idx+=1


    return np.mean(map7)

def variable_transforms(df, date1, date2, time_lag_months):


    product_column_names = list(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'].columns)
    df_1 = df[(df['date'] == date1)]
    df_2 = df[(df['date'] == date2)][['customer_code'] + product_column_names]
    df_merge = pd.merge(df_1, df_2, how = 'inner', on = 'customer_code', suffixes=('', '_future'))

    X = df_merge.loc[:,'date':'ind_recibo_ult1']
    y_present = df_merge.loc[:,'ind_ahor_fin_ult1':'ind_recibo_ult1']
    y_future = df_merge.loc[:,'ind_ahor_fin_ult1_future':'ind_recibo_ult1_future']
    dy = y_future.values-y_present.values
    y = pd.DataFrame(np.where(dy>=0, dy, 0), columns=y_present.columns)

    si_no_dict = {'S':1, 'N': 0}

    X.replace({'is_foreigner': si_no_dict,
              'is_local_resident': si_no_dict,
              'is_spouse_of_employee': si_no_dict,
              'is_deceased': si_no_dict,
              },
              inplace = True)

    # Consolidate labels in some categorical fields.
    # Ex: 1, 1.0, '1.0' are almost certainly entry anomalies and need to be grouped.
    # For categorical fields with only 2 values, replace with 0 or 1. No need to one-hot encode.

    X['gender'] = np.where(X['gender'] == 'V', 1, 0)
    # X['age'] = np.where(X['age'] == ' NA', 0, X['age'])
    # X['age'] = X['age'].astype(int)
    X['customer_seniority'] = pd.to_numeric(X['customer_seniority'], errors = 'coerce')
    X['age'] = pd.to_numeric(X['age'], errors = 'coerce')
    # cust_type_dict = {'P': 5, '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4}
    # X['customer_type_1m'] = X['customer_type_1m'].fillna(value = 0)
    # X.replace({'customer_type_1m': cust_type_dict}, inplace = True)
    # X['customer_type_1m'] = X['customer_type_1m'].astype(int)

    # 99.5% of country_of_residence is Spain. One-hot encoding all of Europe will probably only introduce noise.
    X['local_residence'] = np.where(X['country_of_residence'] == 'ES', 1, 0)

    # keep top 10 province codes. Aggregate remaining codes into "other" category
    top10_prov_code = list(X['province_code'].value_counts().index[:10])
    X['province_code'] = np.where(X['province_code'].isin(top10_prov_code), X['province_code'], -99)
    X['province_code'] = X['province_code'].astype(object)
    X['income_isnull'] = X['gross_household_income'].isnull()

    # get products owned in the last X months
    lag_cutoff_date = pd.to_datetime(date1)-np.timedelta64(time_lag_months, 'M')
    product_cols = list(df.loc[:,'ind_ahor_fin_ult1':'ind_recibo_ult1'].columns)
    df_products = df[['customer_code']+['date']+product_cols]
    df_products = df_products[(df_products['date'] > lag_cutoff_date)
                         & (df_products['date'] <  pd.to_datetime(date1))]
    df_products.drop('date', inplace=True, axis = 1)
    df_products = df_products.groupby('customer_code').max()
    #df_products = y.add_suffix('_future')
    #X = pd.concat([X, df_products.add_suffix('_past_owned')], axis=1)
    #df_products = df_products.add_suffix('_past_owned')

    df_products['customer_code'] = df_products.index
    #X = pd.concat([X, df_products.add_suffix('_past_owned')], axis=1)
    #print(X[product_column_names].shape)
    X = pd.merge(X, df_products, how = 'left', on = 'customer_code', suffixes=('', '_past_owned'))

    for col in product_column_names:
        X[col+'_past_owned'] = X[col+'_past_owned'] - X[col]
        X[col+'_past_owned'] = np.where(X[col+'_past_owned'] < 0, 0, X[col+'_past_owned'])

    return X, y

def variable_transforms_no_future(df, df_past, date, time_lag_months):

    product_column_names = list(df_past.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'].columns)
    X = df[(df['date'] == date)]

    si_no_dict = {'S':1, 'N': 0}
    X.replace({'is_foreigner': si_no_dict,
              'is_local_resident': si_no_dict,
              'is_spouse_of_employee': si_no_dict,
              'is_deceased': si_no_dict,
              },
              inplace = True)

    # Consolidate labels in some categorical fields.
    # Ex: 1, 1.0, '1.0' are almost certainly entry anomalies and need to be grouped.
    # For categorical fields with only 2 values, replace with 0 or 1. No need to one-hot encode.

    X['gender'] = np.where(X['gender'] == 'V', 1, 0)

       # 99.5% of country_of_residence is Spain. One-hot encoding all of Europe will probably only introduce noise.
    X['local_residence'] = np.where(X['country_of_residence'] == 'ES', 1, 0)

    # keep top 10 province codes. Aggregate remaining codes into "other" category
    top10_prov_code = list(X['province_code'].value_counts().index[:10])
    X['province_code'] = np.where(X['province_code'].isin(top10_prov_code), X['province_code'], -99)
    X['income_isnull'] = X['gross_household_income'].isnull()

    # get products owned in the last 6 months
    lag_cutoff_date = pd.to_datetime(date)-np.timedelta64(6, 'M')
    product_cols = list(df_past.loc[:,'ind_ahor_fin_ult1':'ind_recibo_ult1'].columns)
    df_products = df_past[['customer_code']+['date']+product_cols]
    df_products = df_products[(df_products['date'] > lag_cutoff_date)
                         & (df_products['date'] <  pd.to_datetime(date))]
    df_products.drop('date', inplace=True, axis = 1)
    df_products = df_products.groupby('customer_code').max()
    df_products = df_products.add_suffix('_past_owned')
    df_products['customer_code'] = df_products.index
    #X = pd.concat([X, df_products.add_suffix('_past_owned')], axis=1)
    X = pd.merge(X, df_products, how='left', on='customer_code')

    # get products owned only in the previous month:
    #df['YearMonth'] = df['ArrivalDate'].map(lambda x: 100*x.year + x.month)

    lag_cutoff_date = pd.to_datetime(date)-np.timedelta64(1, 'M')
    lag_cutoff_date = lag_cutoff_date.year*100 + lag_cutoff_date.month
    X_lastmonth = df_past[(df_past['date'].map(lambda x: 100*x.year + x.month) == lag_cutoff_date)]

    #X_lastmonth[product_cols] = X_lastmonth[product_cols].add_suffix('')

    X = pd.merge(X, X_lastmonth[['customer_code'] + product_cols], how='left', on='customer_code')

    for col in product_column_names:
        X[col+'_past_owned'] = X[col+'_past_owned'] - X[col]
        X[col+'_past_owned'] = np.where(X[col+'_past_owned'] < 0, 0, X[col+'_past_owned'])

    X['gross_household_income'] = pd.to_numeric(X['gross_household_income'], errors = 'coerce')



    return X
