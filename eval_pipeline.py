from transformers import AutoformerConfig, AutoformerForPrediction
from accelerate import Accelerator
from gluonts.dataset.repository.datasets import get_dataset
import numpy as np
from tqdm.autonotebook import tqdm
from evaluate import load
from gluonts.time_feature import get_seasonality
import matplotlib.dates as mdates
import pandas as pd
import dataloader as dl
import matplotlib.pyplot as plt
from gluonts.dataset.field_names import FieldName



def load_model():
    # Load the pre-trained model autoformer-traffic-hourly
    config = AutoformerConfig.from_pretrained("kashif/autoformer-traffic-hourly")
    model = AutoformerForPrediction.from_pretrained("kashif/autoformer-traffic-hourly")

    return model, config

def forecast_pipeline():

    # get the dataset eg Traffic
    print("Getting Traffic Dataset...")
    dataset = get_dataset("traffic")

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

def eval_():
    forecasts, prediction_length, test_dataset ,_ , freq= forecast_pipeline()

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
    test_ds = list(test_dataset)
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=test_ds[ts_index][FieldName.START],
        periods=len(test_ds[ts_index][FieldName.TARGET]),
        freq=test_ds[ts_index][FieldName.START].freq,
    ).to_timestamp()

    ax.plot(
        index[-5*prediction_length:], 
        test_ds[ts_index]["target"][-5*prediction_length:],
        label="actual",
    )

    plt.plot(
        index[-prediction_length:], 
        np.median(forecasts[ts_index], axis=0),
        label="median",
    )
    
    plt.gcf().autofmt_xdate()
    plt.legend(loc="best")
    plt.show()






