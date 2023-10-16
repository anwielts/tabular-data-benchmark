'''Class for loading tabular datasets from various sources.'''

from sklearn.datasets import fetch_openml


class TabularDataLoader:
    def __init__(self,
                 datasets: list,
                 output_format='pandas',
                 feature_target_splitting: bool =True,
                 ):
        self.feature_target_splitting = feature_target_splitting
        self.datasets = datasets
        self.output_format = output_format

        self.features = []
        self.targets = []

        for dataset in datasets:
            x, y = self.load_tabular_data(dataset)
            self.features.append(x)
            self.targets.append(y)

    def load_tabular_data(self, dataset_name: str):
        features, targets = fetch_openml(
            dataset_name, version=1, as_frame=True, 
            return_X_y=self.feature_target_splitting, 
            parser=self.output_format
        )

        return features, targets
