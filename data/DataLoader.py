'''Class for loading tabular datasets from various sources.'''

from sklearn.datasets import fetch_openml


class TabularDataLoader:
    def __init__(self,
                 feature_target_splitting=True,
                 ):
        self.feature_target_splitting = feature_target_splitting

        self.features = None
        self.targets = None

    def load_tabular_data(self):
        self.features, self.targets = fetch_openml(
            "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
        )
