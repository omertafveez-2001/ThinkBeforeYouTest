from transformers import AutoformerConfig, AutoformerForPrediction
from accelerate import Accelerator
from gluonts.dataset.repository.datasets import get_dataset
import numpy as np
from tqdm.autonotebook import tqdm
from evaluate import load
from gluonts.time_feature import get_seasonality
import pandas as pd
import dataloader as dl
import matplotlib.pyplot as plt
from gluonts.dataset.field_names import FieldName



def load_model():
    # Load the pre-trained model autoformer-traffic-hourly
    config = AutoformerConfig.from_pretrained("kashif/autoformer-traffic-hourly")
    model = AutoformerForPrediction.from_pretrained("kashif/autoformer-traffic-hourly")

    return model, config

def forecast_pipeline(dataset_name="traffic"):
    """
    Forecasting Pipeline for predicting the forecast.

    All the features: shape = (B, features)

    Arguments for the model.generate():

    static_cateogorical_features: Categorical features eg Store ID, product type that do not change over time.
    static_real_features: Real valued features, store size, location score that are static across timesteps.

    past_time_features: Time-dependent covariates that are known in the past for eg hour of the day, day of the week, holiday indicator. 
    These help in model detecting seasonality or calender based trends.

    past_values: Actual target values (sales, energy load) observed in the past. This is the input sequence.
    future_time_features: Time features for the prediction horizon (future day of the work)
    helps the model know what future time context looks like.
    
    past_observed_mask: Binary mask (1 for observed, 0 for missing) that tells the model which past target values are actually valid.

    Data Shape Analysis:
    -> forecasts[0].shape = (64, 100, 24) -> 64 number of time series in a batch (from dataloader).
    100 number of samples per forecast. 24 is the prediction_length -> we are forecasting 24 time steps into the future. (next 24 hours for eg)

    This means for each of the 64 time series, the model gives 100 different 24 step predictions - one for each sample.

    In the dataloaders, we applied rolling windows which is a resampling technique where we train a forecasting model multiple times, each time shifting the window
    forward in time. 
    7 rolling windows -> we make 7 forecasts each from a different point in time.

    After stacking: 7 batches forecasts for 862 time series -> 6034 time series. 
    (6034, 100, 24) -> for each timeseries we are generating 100 different forecasts over the next 24 timesteps.
    """

    # get the dataset eg Traffic
    print(f"Getting {dataset_name} Dataset...")
    dataset = get_dataset(dataset_name)

    print("Dataset retrived...")
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length


    # Split the dataset
    train_dataset = dataset.train
    test_dataset = dataset.test
    print("Dataset split into train and test...")

    # load the model
    print("Load the model and config from Autoformer Transformers Implementation...")
    model, config = load_model()

    # use the create_backtest from dataloader.py to make test_dataloader.
    test_dataloader = dl.create_backtest_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
    )
    

    # Create the accelerator for distributed training
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)
    model.eval()

    # Prediction: forecasts
    print("Computing predictions for the dataset...")
    forecasts_ = []
    for batch in test_dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts_.append(outputs.sequences.cpu().numpy())
    
    print(f"Shape of the forecasts: {forecasts_[0].shape}")
    
    forecasts = np.vstack(forecasts_)
    print(f"Stacked Forecast for all the batches in the dataloader: {forecasts.shape}")

    return forecasts, prediction_length, test_dataset, train_dataset, freq

def eval_(dataset_name="traffic"):
    """
    Eval Pipeline for evaluating the forecasts on MASE metric
    
    Metric Compute Arguments:

    forecast_mediun[item_id]: predicted sequence for a single time-series over the forecast horizon.
    np.array(ground_truth): the actual target values for the last prediction_length time steps in each series.

    training_data: the historical target values prior to the forecast window. MASE uses this to compute a scaling factor
    making the metric scale-independent.

    periodicity: Seasonal Period for the naive benchmark. For example "H" -> hourly, 'D" -> 7. It determines how the naive seasonal
    forecast is constructed (eg lagging by seasonality steps).
    """

    forecasts, prediction_length, test_dataset ,_ , freq= forecast_pipeline(dataset_name=dataset_name)

    print("Evaluaing on MASE metric...")
    mase_metric = load("evaluate-metric/mase")

    forecast_median = np.median(forecasts, 1)

    mase_metrics = []
    for item_id, ts in enumerate(tqdm(test_dataset)):
        training_data = ts["target"][:-prediction_length]
        ground_truth = ts["target"][-prediction_length:]
        mase = mase_metric.compute(
            predictions=forecast_median[item_id], 
            references=np.array(ground_truth), 
            training=np.array(training_data), 
            periodicity=get_seasonality(freq))
        mase_metrics.append(mase["mase"])
    
    print(f"Autoformer univariate MASE: {np.mean(mase_metrics):.3f}")

    return forecasts, prediction_length, test_dataset



def plot(ts_index, test_dataset, prediction_length, forecasts):
    """
    Input params:
    ts_index: index of the timeseries (eg. from 0-6033)
    test_dataset
    prediction_length: Number of time-steps into the future.
    forecasts
    """

    # convert dataset to list
    test_ds = list(test_dataset)

    # set up the plot.
    fig, ax = plt.subplots()

    # construct time index aligned with the full length of the target series
    # starts with the indexed series start and converts periodindex to Timestamp.
    index = pd.period_range(
        start=test_ds[ts_index][FieldName.START],
        periods=len(test_ds[ts_index][FieldName.TARGET]),
        freq=test_ds[ts_index][FieldName.START].freq, # for eg 'H', 'D'
    ).to_timestamp()


    # Plot actual values (last 5 * prediction length)
    ax.plot(
        index[-5*prediction_length:], 
        test_ds[ts_index]["target"][-5*prediction_length:],
        label="actual",
    )

    # plot the median of the samples forecasted series.
    plt.plot(
        index[-prediction_length:], 
        np.median(forecasts[ts_index], axis=0),
        label="median",
    )
    
    plt.gcf().autofmt_xdate()
    plt.legend(loc="best")
    plt.show()






