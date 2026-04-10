import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from typing import List, Tuple, Union
from datetime import datetime
from .utils import extend_time_df

#
import logging
logger = logging.getLogger(__name__)

class TimeSeries():
    def __init__(self, name:str):
        self.is_trained = False
        self.name = name
        self.verbose = True
        self.group = None

    def load_signal(self,
        data: pd.DataFrame,
        target_vars:List[str]=[],
        group: Union[None,str] = None,
        #
        enrich_cat:List[str] = [],
        past_vars:List[str]=[],
        future_vars:List[str]=[],
        categorical_past_vars:List[str]=[],
        categorical_fut_vars:List[str]=[],
        check_past:bool=True,
        check_holes_and_duplicates: bool = True,
        silly_model: bool = False,
        sampler_weights: Union[None,str] = None
        ):
        """
        > data has to have 'time' as timestamp column
        > Checks:
            - Duplicates
            - Frequency inferred
            - Filled Dataset in missing timestamps
        """
        breakpoint()
        dataset = data.copy()
        dataset.sort_values(by='time',inplace=True)                    
        assert len(target_variables)>0, 'Provide at least one column for target'
        assert 'time' in dataset.columns, 'The temporal column must be called time'

        #
        # self.target_vars = list(set(target_vars))
        # self.past_vars = list(set(past_vars))
        # self.future_vars = list(set(future_vars))
        # self.categorical_past_vars = list(set(categorical_past_vars))
        # self.categorical_fut_vars = list(set(categorical_fut_vars))

        #
        self.group = group
        if check_holes_and_duplicates:
            dataset = self._check_holes_and_duplicates(dataset, group)
        else:
            logger.info("I will compute the frequency as minimum of the time difference")
            self.freq = dataset.time.diff().dropna().min()
            if isinstance(dataset.time.dtype, datetime):
                self.freq = pd.to_timedelta(self.freq)

        if set(target_vars).intersection(set(past_variables))!= set(target_vars): 
            if check_past:
                logger.info('I will update past column adding all target columns, if you want to avoid this beahviour please use check_pass as false')
                past_variables = list(set(past_variables).union(set(target_variables)))
                past_variables = list(np.sort(past_variables))

        self.cat_past_var = cat_past_var
        self.cat_fut_var = cat_fut_var

    def _check_holes_and_duplicates(self, dataset, group):
        dataset.drop_duplicates(subset=['time'] if group is None else [group,'time'],  keep='first', inplace=True, ignore_index=True)
        
        if group is None:
            differences = dataset.time.diff().dropna()
        else:
            differences = dataset.groupby(group)['time'].diff().dropna()
        
        min_diff = differences.min()
        if is_datetime64_any_dtype(dataset['time']):
            self.freq = pd.to_timedelta(min_diff)
        elif is_numeric_dtype(dataset['time']):
            self.freq = int(min_diff) 
        else:
            raise TypeError("The 'time' column must be of integer or datetime type.")

        if differences.nunique()>1:
            logger.info("There are holes in the dataset i will try to extend the dataframe inserting NAN")
            logger.info(f'Detected minumum frequency: {freq}')
            dataset = extend_time_df(dataset,freq,group).merge(dataset,how='left')
        return dataset
        
        