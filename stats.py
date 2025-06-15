import ruptures as rpt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import shapiro, levene
from statsmodels.stats.diagnostic import acorr_ljungbox
from joblib import Parallel, delayed


def transforming_ground_truth(dataset, prediction_length):
    ground_truth = []
    for ts in tqdm(dataset):
        target = ts["target"]
        gt = target[-prediction_length:]
        ground_truth.append(np.array(gt))

    ground_truth = np.stack(ground_truth)
    print(f"Ground Truth Shape: {ground_truth.shape}")

    return ground_truth


def compute_bic(signal, n_bkps_range, model="l2"):
    """
    Compute BIC for different numbers of breakpoints and return the optimal breakpoints
    """
    n = len(signal)
    costs = []
    
    for k in n_bkps_range:
        try:
            # Step 1: Segment the signal using Binary Segmentation
            algo = rpt.Binseg(model=model).fit(signal)
            bkps = algo.predict(n_bkps=k)
            
            # Step 2: Get the cost correctly - no need to instantiate manually
            # The algo object already has the fitted cost function
            cost = algo.cost.sum_of_costs(bkps)
            
            # Step 3: Compute BIC
            # BIC = cost + k * log(n) * penalty_factor
            # For time series, we often use a penalty factor
            penalty = k * np.log(n)
            bic = cost + penalty
            
            costs.append((k, bic, bkps))
            
        except Exception as e:
            print(f"Error for k={k}: {e}")
            # If segmentation fails, assign a high BIC
            costs.append((k, np.inf, [n]))
    
    # Step 4: Choose the number of breakpoints with the lowest BIC
    if costs:
        best_k, best_bic, best_bkps = min(costs, key=lambda x: x[1])
        return best_bkps
    else:
        return [n]  # Return just the endpoint if all failed


def compute_bic_alternative(signal, n_bkps_range, model="l2"):
    """
    Alternative implementation using a different approach
    """
    n = len(signal)
    costs = []
    
    # Fit the algorithm once
    algo = rpt.Binseg(model=model).fit(signal)
    
    for k in n_bkps_range:
        try:
            # Get breakpoints
            bkps = algo.predict(n_bkps=k)
            
            # Calculate cost manually using the segments
            cost = 0
            start = 0
            for end in bkps:
                if end > start:
                    segment = signal[start:end]
                    if len(segment) > 0:
                        # For L2 cost, this is the sum of squared deviations from mean
                        segment_mean = np.mean(segment)
                        segment_cost = np.sum((segment - segment_mean) ** 2)
                        cost += segment_cost
                start = end
            
            # BIC = cost + penalty
            bic = cost + k * np.log(n)
            costs.append((k, bic, bkps))
            
        except Exception as e:
            print(f"Error for k={k}: {e}")
            costs.append((k, np.inf, [n]))
    
    if costs:
        best_k, best_bic, best_bkps = min(costs, key=lambda x: x[1])
        return best_bkps
    else:
        return [n]


def run_bic_selection(signal):
    """Wrapper function for parallel processing"""
    return compute_bic(signal, n_bkps_range=range(1, 6), model="l2")

def run_bic_selection_alternative(signal):
    """Alternative wrapper function"""
    return compute_bic_alternative(signal, n_bkps_range=range(1, 6), model="l2")


def structural_breaks(forecast, dataset, prediction_length):
    # Run for all 24 hours
    downsample_factor = 10
    mean_forecast = forecast.mean(axis=1)

    ground_truth = transforming_ground_truth(dataset, prediction_length)

    mean_forecast_ds = mean_forecast[::downsample_factor]  # (603, 24)
    ground_truth_ds = ground_truth[::downsample_factor]

    # Try the main approach first, fall back to alternative if needed
    try:
        print("Running BIC computation with primary method...")
        # Test with a single signal first
        test_signal = mean_forecast_ds[:, 0]
        test_result = run_bic_selection(test_signal)
        print(f"Test successful. First breakpoints: {test_result}")
        
        # If test passes, run parallel computation
        breaks_per_hour = Parallel(n_jobs=-1)(
            delayed(run_bic_selection)(mean_forecast_ds[:, t]) for t in range(24)
        )
        gt_breaks_per_hour = Parallel(n_jobs=-1)(
            delayed(run_bic_selection)(ground_truth_ds[:, t]) for t in range(24)
        )
        
    except Exception as e:
        print(f"Primary method failed: {e}")
        print("Trying alternative method...")
        
        # Use alternative method
        breaks_per_hour = Parallel(n_jobs=-1)(
            delayed(run_bic_selection_alternative)(mean_forecast_ds[:, t]) for t in range(24)
        )
        gt_breaks_per_hour = Parallel(n_jobs=-1)(
            delayed(run_bic_selection_alternative)(ground_truth_ds[:, t]) for t in range(24)
        )

    # Build breakpoint matrices
    T = mean_forecast_ds.shape[0]
    break_matrix_pred = np.zeros((24, T))
    break_matrix_true = np.zeros((24, T))

    for t in range(24):
        # Remove the last breakpoint (which is always the endpoint)
        pred_breaks = breaks_per_hour[t][:-1] if len(breaks_per_hour[t]) > 1 else []
        true_breaks = gt_breaks_per_hour[t][:-1] if len(gt_breaks_per_hour[t]) > 1 else []
        
        for b in pred_breaks:
            if 0 <= b < T:  # Ensure breakpoint is within bounds
                break_matrix_pred[t, b] = 1
        
        for b in true_breaks:
            if 0 <= b < T:  # Ensure breakpoint is within bounds
                break_matrix_true[t, b] = 1
    
    return break_matrix_pred, break_matrix_true, mean_forecast_ds, ground_truth_ds, breaks_per_hour, gt_breaks_per_hour



# Define a function to compute all 3 tests for a single series ---
def analyze_series(i, forecast):
    try:
        # Normality: Shapiro-Wilk on each of the 24 forecast hours
        norm_pvals = [shapiro(forecast[i, :, t])[1] for t in range(24)]
        normality_score = np.mean(np.array(norm_pvals) > 0.05)

        # Heteroscedasticity: Levene test on hour-to-hour forecast distributions
        hetero_pvals = [levene(forecast[i, :, t - 1], forecast[i, :, t])[1] for t in range(1, 24)]
        hetero_score = np.mean(np.array(hetero_pvals) > 0.05)

        # Autocorrelation: Ljung-Box test on the mean prediction series
        lb_test = acorr_ljungbox(forecast[i].mean(axis=0), lags=[10], return_df=True)
        autocorr_score = float(lb_test["lb_pvalue"].iloc[0] > 0.05)

        # Combine into a trust score
        return (normality_score + hetero_score + autocorr_score) / 3
    except Exception as e:
        print(f"Error at series {i}: {e}")
        return 0.0  # fallback in case of errors

def trust_score(forecasts):
    trust_score = Parallel(n_jobs=-1)(
    delayed(analyze_series)(i, forecast) for i in range(forecasts.shape[0])
    )
    trust_score = np.array(trust_score)

    return trust_score



