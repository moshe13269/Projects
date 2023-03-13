from torch import nn
from typing import List, Tuple


class LinearClassifier(nn.Module):
    """
    Transformer Encdoer outputs are: (batch, t', channels)
    Given N classes (types of parameters) when all specific class contain num_classes
    The Dense layer mapping latent space to accurate outputs:
    outputs -> class_i: (,t', channels) -> (, num_classes) and softmax

    """

    outputs_dimension_per_outputs: List[int]
    activation: str = 'softmax'
    dropout: float = 0.2

    def __init__(self,
                 outputs_dimension_per_outputs,
                 # num_classes_per_param: List[int],
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.activation = activation
        self.layers = nn.Sequential(
            # tf.keras.layers.GlobalAveragePooling1D(), #Flatten(),
            nn.Flatten(),
            nn.Linear(in_features=65*512, out_features=sum(outputs_dimension_per_outputs)), #Dense(units=512 * 50, activation=self.activation),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=sum(outputs_dimension_per_outputs),
                      out_features=sum(outputs_dimension_per_outputs)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=sum(outputs_dimension_per_outputs),
                      out_features=sum(outputs_dimension_per_outputs)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=sum(outputs_dimension_per_outputs),
                      out_features=sum(outputs_dimension_per_outputs)),
            nn.ReLU()
        )

    def forward(self, x):
        outputs = self.layers(x)

        return outputs
