
import os
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix


class Results:

    path2save_results: str
    num_class: int

    def __init__(self, path2save_results: str,
                 num_class: int):
        self.path2save_results = path2save_results
        self.num_class = num_class

    def csv_predicted(self, model, test_set):
        num_sample = int(test_set.__len__().numpy())
        test_set = test_set.as_numpy_iterator()

        results = np.zeros((num_sample * 2, 16))

        for sample in range(num_sample):
            x, y = test_set.next()
            y_ = model.predict_on_batch(x)[0]  # model.predict(x)
            results[2 * sample: 2 * sample + 1, :] = y_.squeeze()
            results[2 * sample + 1: 2 * sample + 2, :] = y[0].squeeze()

        pd.DataFrame(results).to_csv(os.path.join(self.path2save_results, 'csv_results.csv'))

    def plot_ROC_OvR(self):
        pass

    def plot_ROC_OvO(self):
        pass

    def confusion_matrix(self, y_true, y_pred, labels, normalize='all'):
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels, normalize=normalize)

        # change each element of z to type string for annotations
        confusion_matrix_text = [[str(labels) for labels in labels] for labels in cm]

        # set up figure
        fig = ff.create_annotated_heatmap(cm,
                                          x=labels,
                                          y=labels,
                                          annotation_text=confusion_matrix_text,
                                          colorscale='Viridis')

        # add title
        fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                          # xaxis = dict(title='x'),
                          # yaxis = dict(title='x')
                          )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.35,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig['data'][0]['showscale'] = True
        fig.write_html(self.path2save_results + 'confusion_matrix' + '.html')
        # fig.show()




