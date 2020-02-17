import os

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

import pathlib
import joblib

from src.config import config
from src.modeling import transformers

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables            
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit statement to accommodate the sklearn pipeline."""

        return self
            
    def transform(self, X):
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X 
    
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')

        return X
    
      
class LabelEncoderExt(object):
    def __init__(self):
        """It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id.
        """
        self.label_encoder = preprocessing.LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """This will fit the encoder for all the unique values and introduce unknown value."""
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """This will transform the data_list to id list where the new values get assigned to Unknown class."""
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables=None, method='mode'):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables            
        self.method = method

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            if self.method == 'median':
                self.imputer_dict_[feature] = X[feature].median()
            elif self.method == 'mean':
                self.imputer_dict_[feature] = X[feature].mean()
            else:
                self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
    

class LabelEncoders(BaseEstimator, TransformerMixin):
    """Encoders for each (categorical)_variables."""
    
    def __init__ (self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        X = X.copy()
        
        # persist a dictionary of encoders
        self.encoder_dict_ = {}                      
        
        for feature in self.variables:
            self.encoder_dict_[feature] = LabelEncoderExt()
            self.encoder_dict_[feature].fit(X[feature])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = self.encoder_dict_[feature].transform(X[feature])
        return X 