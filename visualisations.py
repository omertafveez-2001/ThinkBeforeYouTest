import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.field_names import FieldName
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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


def visualise_structuralbreaks(break_matrix_pred, break_matrix_true):
    # Visualize
    fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Plot predicted breaks
    im1 = axs[0].imshow(break_matrix_pred, aspect='auto', cmap='Reds', interpolation='nearest')
    axs[0].set_title("Predicted Structural Breaks (per hour)")
    axs[0].set_ylabel("Hour of Day")

    # Plot true breaks
    im2 = axs[1].imshow(break_matrix_true, aspect='auto', cmap='Blues', interpolation='nearest')
    axs[1].set_title("True Structural Breaks (per hour)")
    axs[1].set_ylabel("Hour of Day")
    axs[1].set_xlabel("Downsampled Time Series Index")

    # Add colorbars
    plt.colorbar(im1, ax=axs[0], label='Break Present')
    plt.colorbar(im2, ax=axs[1], label='Break Present')

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Total predicted breaks: {np.sum(break_matrix_pred)}")
    print(f"Total true breaks: {np.sum(break_matrix_true)}")
    print(f"Breaks per hour (predicted): {np.sum(break_matrix_pred, axis=1)}")
    print(f"Breaks per hour (true): {np.sum(break_matrix_true, axis=1)}")



def create_comprehensive_break_analysis(mean_forecast_ds, ground_truth_ds, 
                                      breaks_per_hour, gt_breaks_per_hour, 
                                      downsample_factor=10):
    """
    Create comprehensive visualizations for structural break analysis
    """
    T, H = mean_forecast_ds.shape  # Time points, Hours
    
    # 1. OVERLAY TIME SERIES WITH BREAKS (keep as subplots)
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    fig.suptitle('Time Series with Structural Breaks (First 24 Hours)', fontsize=16)
    
    for h in range(min(24, H)):
        row, col = h // 6, h % 6
        ax = axes[row, col]
        
        # Plot time series
        time_idx = np.arange(T) * downsample_factor
        ax.plot(time_idx, mean_forecast_ds[:, h], 'b-', alpha=0.7, label='Forecast', linewidth=1)
        ax.plot(time_idx, ground_truth_ds[:, h], 'r-', alpha=0.7, label='Ground Truth', linewidth=1)
        
        # Add predicted breaks as vertical lines
        pred_breaks = [b for b in breaks_per_hour[h][:-1] if 0 <= b < T]
        for b in pred_breaks:
            ax.axvline(x=b*downsample_factor, color='blue', alpha=0.8, linestyle='--', linewidth=2)
        
        # Add true breaks as vertical lines
        true_breaks = [b for b in gt_breaks_per_hour[h][:-1] if 0 <= b < T]
        for b in true_breaks:
            ax.axvline(x=b*downsample_factor, color='red', alpha=0.8, linestyle=':', linewidth=2)
        
        ax.set_title(f'Hour {h}', fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        if h == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 2. BREAK DENSITY HEATMAP WITH STATISTICS (now separate figures)
    
    # Create break matrices
    break_matrix_pred = np.zeros((H, T))
    break_matrix_true = np.zeros((H, T))
    
    for h in range(H):
        for b in breaks_per_hour[h][:-1]:
            if 0 <= b < T:
                break_matrix_pred[h, b] = 1
        for b in gt_breaks_per_hour[h][:-1]:
            if 0 <= b < T:
                break_matrix_true[h, b] = 1
    
    # Heatmap of predicted breaks
    plt.figure(figsize=(12, 8))
    sns.heatmap(break_matrix_pred, cmap='Reds', cbar_kws={'label': 'Break Present'}, xticklabels=False)
    plt.title('Predicted Structural Breaks by Hour')
    plt.ylabel('Hour of Day')
    plt.xlabel('Time Index')
    plt.show()
    
    # Heatmap of true breaks
    plt.figure(figsize=(12, 8))
    sns.heatmap(break_matrix_true, cmap='Blues', cbar_kws={'label': 'Break Present'}, xticklabels=False)
    plt.title('True Structural Breaks by Hour')
    plt.ylabel('Hour of Day')
    plt.xlabel('Time Index')
    plt.show()
    
    # Break frequency by hour
    pred_freq = np.sum(break_matrix_pred, axis=1)
    true_freq = np.sum(break_matrix_true, axis=1)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(H)
    width = 0.35
    plt.bar(x - width/2, pred_freq, width, label='Predicted', alpha=0.8, color='red')
    plt.bar(x + width/2, true_freq, width, label='True', alpha=0.8, color='blue')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Breaks')
    plt.title('Break Frequency by Hour')
    plt.legend()
    plt.xticks(range(0, H, 4))
    plt.show()
    
    # Break timing distribution
    pred_times = [b for h in range(H) for b in breaks_per_hour[h][:-1] if 0 <= b < T]
    true_times = [b for h in range(H) for b in gt_breaks_per_hour[h][:-1] if 0 <= b < T]
    
    plt.figure(figsize=(12, 6))
    plt.hist(pred_times, bins=50, alpha=0.6, label='Predicted', color='red', density=True)
    plt.hist(true_times, bins=50, alpha=0.6, label='True', color='blue', density=True)
    plt.xlabel('Time Index')
    plt.ylabel('Density')
    plt.title('Break Timing Distribution')
    plt.legend()
    plt.show()
    
    # 3. SEGMENT ANALYSIS (now separate figures)
    def analyze_segments(data, breaks_list):
        """Analyze segments between breaks"""
        segments_info = []
        for h in range(len(breaks_list)):
            breaks = [0] + breaks_list[h]  # Add start point
            for i in range(len(breaks)-1):
                start, end = breaks[i], breaks[i+1]
                if end > start and end <= len(data):
                    segment = data[start:end, h]
                    segments_info.append({
                        'hour': h,
                        'start': start,
                        'end': end,
                        'length': end - start,
                        'mean': np.mean(segment),
                        'std': np.std(segment),
                        'trend': np.polyfit(range(len(segment)), segment, 1)[0] if len(segment) > 1 else 0
                    })
        return segments_info
    
    pred_segments = analyze_segments(mean_forecast_ds, breaks_per_hour)
    true_segments = analyze_segments(ground_truth_ds, gt_breaks_per_hour)
    
    # Segment lengths
    pred_lengths = [s['length'] for s in pred_segments]
    true_lengths = [s['length'] for s in true_segments]
    
    plt.figure(figsize=(10, 6))
    plt.hist(pred_lengths, bins=20, alpha=0.6, label='Predicted', color='red')
    plt.hist(true_lengths, bins=20, alpha=0.6, label='True', color='blue')
    plt.xlabel('Segment Length')
    plt.ylabel('Frequency')
    plt.title('Segment Length Distribution')
    plt.legend()
    plt.show()
    
    # Segment means
    pred_means = [s['mean'] for s in pred_segments]
    true_means = [s['mean'] for s in true_segments]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(pred_means)), pred_means, alpha=0.6, label='Predicted', color='red', s=20)
    plt.scatter(range(len(true_means)), true_means, alpha=0.6, label='True', color='blue', s=20)
    plt.xlabel('Segment Index')
    plt.ylabel('Segment Mean')
    plt.title('Segment Mean Values')
    plt.legend()
    plt.show()
    
    # Segment volatility (std)
    pred_stds = [s['std'] for s in pred_segments]
    true_stds = [s['std'] for s in true_segments]
    
    plt.figure(figsize=(10, 6))
    plt.hist(pred_stds, bins=20, alpha=0.6, label='Predicted', color='red')
    plt.hist(true_stds, bins=20, alpha=0.6, label='True', color='blue')
    plt.xlabel('Segment Std Dev')
    plt.ylabel('Frequency')
    plt.title('Segment Volatility Distribution')
    plt.legend()
    plt.show()
    
    # Segment trends
    pred_trends = [s['trend'] for s in pred_segments]
    true_trends = [s['trend'] for s in true_segments]
    
    plt.figure(figsize=(10, 6))
    plt.hist(pred_trends, bins=20, alpha=0.6, label='Predicted', color='red')
    plt.hist(true_trends, bins=20, alpha=0.6, label='True', color='blue')
    plt.xlabel('Segment Trend (slope)')
    plt.ylabel('Frequency')
    plt.title('Segment Trend Distribution')
    plt.legend()
    plt.show()
    
    # Mean vs Volatility scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_means, pred_stds, alpha=0.6, label='Predicted', color='red', s=30)
    plt.scatter(true_means, true_stds, alpha=0.6, label='True', color='blue', s=30)
    plt.xlabel('Segment Mean')
    plt.ylabel('Segment Std Dev')
    plt.title('Mean vs Volatility')
    plt.legend()
    plt.show()
    
    # Segment length vs hour
    pred_hours = [s['hour'] for s in pred_segments]
    true_hours = [s['hour'] for s in true_segments]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_hours, pred_lengths, alpha=0.6, label='Predicted', color='red', s=30)
    plt.scatter(true_hours, true_lengths, alpha=0.6, label='True', color='blue', s=30)
    plt.xlabel('Hour of Day')
    plt.ylabel('Segment Length')
    plt.title('Segment Length by Hour')
    plt.legend()
    plt.show()
    
    # 4. ACCURACY METRICS VISUALIZATION (now separate figures)
    def compute_break_metrics(pred_breaks, true_breaks, tolerance=2):
        """Compute precision, recall, F1 for break detection"""
        metrics_by_hour = []
        
        for h in range(len(pred_breaks)):
            pred = set(pred_breaks[h][:-1])  # Remove endpoint
            true = set(true_breaks[h][:-1])
            
            # True positives with tolerance
            tp = 0
            for p in pred:
                if any(abs(p - t) <= tolerance for t in true):
                    tp += 1
            
            fp = len(pred) - tp
            fn = len(true) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_by_hour.append({
                'hour': h,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'pred_count': len(pred),
                'true_count': len(true)
            })
        
        return metrics_by_hour
    
    metrics = compute_break_metrics(breaks_per_hour, gt_breaks_per_hour)
    
    hours = [m['hour'] for m in metrics]
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    
    # Performance by hour
    plt.figure(figsize=(12, 6))
    plt.plot(hours, precision, 'ro-', label='Precision', markersize=4)
    plt.plot(hours, recall, 'bo-', label='Recall', markersize=4)
    plt.plot(hours, f1, 'go-', label='F1-Score', markersize=4)
    plt.xlabel('Hour of Day')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Hour')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    plt.show()
    
    # Confusion matrix-style visualization
    tp_counts = [m['tp'] for m in metrics]
    fp_counts = [m['fp'] for m in metrics]
    fn_counts = [m['fn'] for m in metrics]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(hours))
    width = 0.25
    
    plt.bar(x - width, tp_counts, width, label='True Positives', color='green', alpha=0.7)
    plt.bar(x, fp_counts, width, label='False Positives', color='red', alpha=0.7)
    plt.bar(x + width, fn_counts, width, label='False Negatives', color='orange', alpha=0.7)
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.title('Break Detection Errors by Hour')
    plt.legend()
    plt.xticks(range(0, len(hours), 4))
    plt.show()
    
    # Overall performance summary
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    plt.figure(figsize=(8, 6))
    performance_data = [avg_precision, avg_recall, avg_f1]
    performance_labels = ['Precision', 'Recall', 'F1-Score']
    colors = ['red', 'blue', 'green']
    
    bars = plt.bar(performance_labels, performance_data, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Overall Performance Summary')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, performance_data):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    plt.show()
    
    # Break count comparison
    pred_counts = [m['pred_count'] for m in metrics]
    true_counts = [m['true_count'] for m in metrics]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(true_counts, pred_counts, alpha=0.7, s=50)
    plt.plot([0, max(max(pred_counts), max(true_counts))], 
             [0, max(max(pred_counts), max(true_counts))], 
             'r--', alpha=0.5, label='Perfect Agreement')
    plt.xlabel('True Break Count')
    plt.ylabel('Predicted Break Count')
    plt.title('Break Count Agreement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return metrics, pred_segments, true_segments

def visualise_trustscore(trust_score):
    plt.figure(figsize=(10, 4))
    plt.hist(trust_score, bins=30, color='purple', edgecolor='black')
    plt.title("Trust Score Distribution across Forecasted Series")
    plt.xlabel("Trust Score (0 = Bad, 1 = High Confidence)")
    plt.ylabel("Number of Time Series")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



