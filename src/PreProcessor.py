class PreProcessor:
    def __init__(self, data):
        self.data = data
        column_groups = self.split_columns()
        self.numerical_cols = column_groups[0]


    def split_columns(self):
        if("label" in self.data.columns):
            data = self.data.drop(columns=["label"])

        numerical_columns = data.columns.tolist()
        return [numerical_columns]
        
