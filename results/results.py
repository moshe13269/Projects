
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix


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




