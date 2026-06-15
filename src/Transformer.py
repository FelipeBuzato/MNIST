from sklearn.base import BaseEstimator, TransformerMixin

class Transformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.features_in_ = X.copy().columns.tolist()
        self.features_out_ = self.transform(X).columns.tolist()
        return self

    
    def transform(self, data):
        ## Implement feature engineering here  
        X = data.copy()

        return X