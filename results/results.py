
class Results:

    path2save_results: str
    num_class: int

    def __init__(self, path2save_results: str,
                 num_class: int):
        self.path2save_results = path2save_results
        self.num_class = num_class

    def plot_ROC_OvR(self):
        pass

    def plot_ROC_OvO(self):
        pass

    def confusion_matrix(self):
        pass






