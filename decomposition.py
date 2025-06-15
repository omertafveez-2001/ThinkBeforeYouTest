import torch
from torch import nn

class DecompositonLayer(nn.Module):
    """
    Returns the trend and the seasonal parts of the time series.
    """

    def __init__(self, kernal_size):
        super().__init__()
        self.kernal_size = kernal_size 
        self.avg = nn.AvgPool1d(kernal_size = kernal_size, stride = 1, padding = 0) # moving average

    def forward(self, x):
        """
        Input shape: Batch x Time x Embed_Dim
        """
        # padding on both ends of time series to match the shape of original series
        # with avg pooling the output will be smaller than the input.
        # so we pad the start and end.
        num_of_pads = (self.kernal_size -1)//2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[: -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, end], dim = 1)

        # calculate the trend and seasonal part of the series
        # conver the shape fo B x D X T since we want temporal smoothig and then rever it back.
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend

        return x_seasonal, x_trend
    


