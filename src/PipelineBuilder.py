from Transformer import Transformer
from PreProcessorSimple import PreProcessorSimple
from PreProcessorNN import PreProcessorNN
from ModelCollection import ModelCollection
from sklearn.pipeline import Pipeline

class PipelineBuilder:
    def __init__(self, data):
        self.data = data
        self.transformer = Transformer()
        self.model_colection = ModelCollection()

    
    def build(self, model_name, params=None):
        model = self.model_colection.get(model_name=model_name, params=params)

        if(model_name in ["KNN", "XGBoost", "My NN"]):
            preprocessor = self.get_preprocessor_simple()
        
        elif(model_name in ["MLP", "NN", "CNN"]):
            preprocessor = self.get_preprocessor_NN()

        pipeline = Pipeline([
            ("transformer", self.transformer),
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        return pipeline
    

    def get_preprocessor_simple(self):
        return PreProcessorSimple(self.data).build()


    def get_preprocessor_NN(self):
        return PreProcessorNN(self.data).build()