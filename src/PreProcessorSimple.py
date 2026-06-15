from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from PreProcessor import PreProcessor

class PreProcessorSimple(PreProcessor):
    def build(self):
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            #("scaler", StandardScaler())
        ])

        column_transformer = ColumnTransformer([
            ("num", numerical_pipeline, self.numerical_cols)
        ])

        column_transformer.set_output(transform="pandas")
        return column_transformer