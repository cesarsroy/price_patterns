
import numpy as np
import stumpy
import pandas as pd


def get_dataframe_with_closest_subsequences(series,indices,m):
    df = pd.DataFrame([])
    last_series = series.iloc[-m:]
    df['last_series'] = last_series.values
    
    for ix in indices:
        data = series.iloc[ix:ix+m].values
        df[series.index[ix].strftime('%Y-%m-%d')] = data
    
    return df

def get_closest_subsequences(series,n=10,m=22):
    '''Gets an array of integers that corresponds to the numerical
    indices of the original series where a close subseuence starts.
    
    Stumpy may give us subsequences that overlap (for example, index 459 and 461)
    so this function changes that by discarding susbsequences where the price 
    may overlap with another.
    
    args:
        series: pandas.Series: The whole series from start to end.
        n: int (optional): The number of closest subsequences the algorithm will extract.
        m: int (optional): The number of data points that constitute the subsequences.
    returns:
        dict: Two keys:
            indices: np.array of indices
            values: The values corresponding to the euclidian distance
            calculated by stumpy
    
    '''
    
    subseq = series.iloc[-m:]  # the last subsequence that we try to match
    seq = series.iloc[:-m]  # the rest of the series
    
    distance_profile = stumpy.mass(subseq,seq) / m
    distance_sorted_ix = np.argsort(distance_profile)

    closest_subsequence_ix = np.zeros(n,dtype=np.int64)
    count = 0
    for ix in distance_sorted_ix:
        is_new_subseq = np.all(np.abs(ix - closest_subsequence_ix) > m)
        if is_new_subseq:
            closest_subsequence_ix[count] = ix
            count += 1
        if count == n:
            break
        
    return dict(indices=closest_subsequence_ix,
                values=distance_profile[closest_subsequence_ix],
                dates=series.index[closest_subsequence_ix])

def get_the_afteraction_subsequences(series,indices,m):
    '''It gets the m-long subsequences after the closest
    subsequences. The price is normalized at a base of 100 '''
    data = pd.DataFrame([])
    for ix in indices:
        after_series = series.iloc[ix+m:ix+m*2].values
        base_100 = 100 * after_series / after_series[0]
        data[series.index[ix].strftime('%Y-%m-%d')] = base_100
    return data

def get_normalized_series(series,indices,m):
    '''Gives back the series normalized using z-scores'''
    df = pd.DataFrame([])
    last_series = series.iloc[-m:].values
    df['last_series'] = (last_series - last_series.mean()) / last_series.std()

    for i,ix in enumerate(indices):
        data = series.iloc[ix:ix+m].values
        normal_data = (data - data.mean()) / data.std()
        df[series.index[ix].strftime('%Y-%m-%d')] = normal_data
    return df
    