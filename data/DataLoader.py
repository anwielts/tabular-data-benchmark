'''Class for loading tabular datasets from various sources.'''

from sklearn.datasets import fetch_openml


TOPIC_DATASET_MAPPING = {
    'physics': ['higgs'],
    'geology': ['covertype'],
    'computer_science': ['letor'],
    'medical': ['breast_cance_wisconsin', 'breast_cance_wisconsin_wdbc'],
    'real_estate': ['california_housning', 'heloc'],
    'income': ['cencus_income']
}


class TabularDataLoader:
    def __init__(self,
                 datasets: list = [],
                 topic: str = '',
                 output_format='pandas',
                 feature_target_splitting: bool =True,
                 ):
        self.feature_target_splitting = feature_target_splitting
        self.datasets = datasets
        self.topic = topic
        self.output_format = output_format

        self.features = []
        self.targets = []

        if len(self.datasets) > 0:
            for dataset in self.datasets:
                x, y = self.load_tabular_data(dataset)
                self.features.append(x)
                self.targets.append(y)
        elif self.topic != '':
            for dataset in TOPIC_DATASET_MAPPING[self.topic]:
                x, y = self.load_tabular_data(dataset)
                self.features.append(x)
                self.targets.append(y)
        else:
            raise Exception

        

    def load_tabular_data(self, dataset_name: str):
        features, targets = fetch_openml(
            dataset_name, version=1, as_frame=True, 
            return_X_y=self.feature_target_splitting, 
            parser=self.output_format
        )

        return features, targets
