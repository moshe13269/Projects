import torch
from torch import nn


class Spectral_LogSTFTMagnitude_Loss(nn.Module):

    def __init__(self, stft=True):
        super(Spectral_LogSTFTMagnitude_Loss, self).__init__()
        self.loss = nn.MSELoss()
        # self.reshape = tf.keras.layers.Reshape((16384,))
        self.mae_loss = nn.L1Loss()
        self.stft = stft

    def forward(self, output, target):
        x_mag, y_mag = torch.split(output, split_size_or_sections=2, dim=0)

        if not self.stft:
            x_mag = torch.abs(
                torch.stft(self.reshape(x_mag), win_length=256, hop_length=128, n_fft=256))

            y_mag = torch.abs(
                torch.stft(self.reshape(y_mag), win_length=256, hop_length=128, n_fft=256))

        x_mag_log = torch.log(x_mag + 1.) #x_mag #
        y_mag_log = torch.log(y_mag + 1.) #y_mag #

        log_stft_magnitude_loss = self.mae_loss(x_mag_log, y_mag_log)

        spectral_convergent_loss = torch.norm((y_mag - x_mag), keepdim=True) / \
            (torch.norm(y_mag, keepdim=True) + 10**-10)
        return log_stft_magnitude_loss + spectral_convergent_loss
