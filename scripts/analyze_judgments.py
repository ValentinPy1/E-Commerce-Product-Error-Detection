from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def load_judgments(judgments_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all judgment CSV files from the directory."""
    judgments = {}
    for csv_file in Path(judgments_dir).glob("*_judgments.csv"):
        model_name = csv_file.stem.replace("_judgments", "")
        df = pd.read_csv(csv_file)
        if not df.empty and "model_conf" in df.columns:
            judgments[model_name] = df
    return judgments


def calculate_confidence_thresholds(judgments: Dict[str, pd.DataFrame], 
                                  min_samples_per_bin: int = 5,
                                  target_accuracy: float = 0.5) -> Dict[str, Dict]:
    """Calculate confidence thresholds for each model."""
    results = {}
    
    for model_name, df in judgments.items():
        print(f"\nAnalyzing {model_name}...")
        
        # Filter valid judgments
        valid = df[df["verdict"].isin(["model", "nature"])].copy()
        print(f"  Valid judgments: {len(valid)}")
        
        if len(valid) < 10:
            print(f"  Skipping {model_name}: insufficient samples ({len(valid)} < 10)")
            continue
            
        # Calculate overall accuracy in disagreements
        model_correct_total = (valid["verdict"] == "model").sum()
        overall_accuracy = model_correct_total / len(valid)
        print(f"  Overall accuracy in disagreements: {overall_accuracy:.3f} ({model_correct_total}/{len(valid)})")
            
        # Create confidence bins
        valid["conf_bin"] = pd.cut(valid["model_conf"], bins=10, include_lowest=True)
        
        # Calculate accuracy per bin
        bin_stats = []
        for bin_name, bin_data in valid.groupby("conf_bin", observed=False):
            if len(bin_data) >= min_samples_per_bin:
                bin_model_correct = (bin_data["verdict"] == "model").sum()
                total = len(bin_data)
                accuracy = bin_model_correct / total
                conf_mean = bin_data["model_conf"].mean()
                bin_stats.append({
                    "conf_bin": bin_name,
                    "conf_mean": conf_mean,
                    "accuracy": accuracy,
                    "count": total,
                    "model_correct": bin_model_correct
                })
                print(f"    Bin {bin_name}: conf={conf_mean:.3f}, accuracy={accuracy:.3f}, count={total}")
        
        if not bin_stats:
            print(f"  Skipping {model_name}: no bins with sufficient samples")
            continue
            
        bin_df = pd.DataFrame(bin_stats)
        print(f"  Bins with sufficient samples: {len(bin_df)}")
        
        # Find threshold where accuracy crosses target_accuracy
        threshold = None
        threshold_method = None
        
        # Method 1: Linear regression threshold
        if len(bin_df) >= 2:
            try:
                from sklearn.linear_model import LinearRegression
                
                X = bin_df["conf_mean"].values.reshape(-1, 1)
                y = bin_df["accuracy"].values
                
                # Weight by sample size
                weights = np.array([stats["count"] for stats in bin_stats])
                
                # Fit weighted linear regression
                reg = LinearRegression()
                reg.fit(X, y, sample_weight=weights)
                
                # Find threshold where regression predicts target_accuracy
                if reg.coef_[0] != 0:  # Avoid division by zero
                    threshold = (target_accuracy - reg.intercept_) / reg.coef_[0]
                    threshold_method = "linear_regression"
                    print(f"  Found threshold via linear regression: {threshold:.3f}")
                    
                    # Check if threshold is within reasonable range
                    if threshold < 0 or threshold > 1:
                        print(f"  Warning: Threshold {threshold:.3f} outside [0,1] range, using linear interpolation")
                        threshold = None
                        threshold_method = None
                        
            except Exception as e:
                print(f"  Linear regression failed: {e}")
        
        # Method 2: Linear interpolation between bins (fallback)
        if threshold is None:
            for i in range(len(bin_df) - 1):
                if (bin_df.iloc[i]["accuracy"] <= target_accuracy and 
                    bin_df.iloc[i + 1]["accuracy"] > target_accuracy):
                    # Linear interpolation
                    acc1, acc2 = bin_df.iloc[i]["accuracy"], bin_df.iloc[i + 1]["accuracy"]
                    conf1, conf2 = bin_df.iloc[i]["conf_mean"], bin_df.iloc[i + 1]["conf_mean"]
                    threshold = conf1 + (target_accuracy - acc1) * (conf2 - conf1) / (acc2 - acc1)
                    threshold_method = "linear_interpolation"
                    print(f"  Found threshold via linear interpolation: {threshold:.3f}")
                    break
        
        # Method 3: Logistic regression if linear methods fail
        if threshold is None and len(bin_df) >= 3:
            try:
                # Fit logistic regression
                X = bin_df["conf_mean"].values.reshape(-1, 1)
                y = (bin_df["accuracy"] > target_accuracy).astype(int)
                
                if len(np.unique(y)) > 1:  # Need both classes
                    from sklearn.linear_model import LogisticRegression
                    lr = LogisticRegression()
                    lr.fit(X, y)
                    
                    # Find threshold where probability = 0.5
                    threshold = -lr.intercept_[0] / lr.coef_[0][0]
                    threshold_method = "logistic_regression"
                    print(f"  Found threshold via logistic regression: {threshold:.3f}")
            except Exception as e:
                print(f"  Logistic regression failed: {e}")
        
        # Method 4: Simple threshold at first bin above target accuracy
        if threshold is None:
            above_target = bin_df[bin_df["accuracy"] > target_accuracy]
            if not above_target.empty:
                threshold = above_target.iloc[0]["conf_mean"]
                threshold_method = "first_above_target"
                print(f"  Found threshold via first above target: {threshold:.3f}")
            else:
                print(f"  No bins above target accuracy {target_accuracy}")
        
        results[model_name] = {
            "threshold": threshold,
            "threshold_method": threshold_method,
            "bin_stats": bin_df,
            "total_samples": len(valid),
            "model_accuracy_overall": overall_accuracy,
            "model_correct_count": model_correct_total
        }
    
    return results


def plot_confidence_analysis(judgments: Dict[str, pd.DataFrame], 
                           thresholds: Dict[str, Dict],
                           output_dir: str):
    """Create plots for confidence analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot all models, not just those with thresholds
    all_models = list(thresholds.keys())
    
    if not all_models:
        print("No models to plot!")
        return pd.DataFrame()
    
    # Overall accuracy vs confidence plot
    n_models = len(all_models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    plt.figure(figsize=(5*cols, 4*rows))
    
    plot_idx = 1
    for model_name in all_models:
        df = judgments[model_name]
        threshold_info = thresholds[model_name]
        
        plt.subplot(rows, cols, plot_idx)
        
        # Filter valid judgments
        valid = df[df["verdict"].isin(["model", "nature"])].copy()
        if len(valid) < 10:
            continue
        
        # Create confidence bins
        valid["conf_bin"] = pd.cut(valid["model_conf"], bins=10, include_lowest=True)
        
        # Calculate accuracy per bin with confidence intervals
        bin_stats = []
        all_confidences = []
        all_accuracies = []
        
        for bin_name, bin_data in valid.groupby("conf_bin", observed=False):
            if len(bin_data) >= 3:
                bin_model_correct = (bin_data["verdict"] == "model").sum()
                total = len(bin_data)
                accuracy = bin_model_correct / total
                conf_mean = bin_data["model_conf"].mean()
                
                # Calculate confidence interval using Wilson score interval
                from scipy import stats
                if total > 0 and accuracy > 0 and accuracy < 1:
                    # Wilson score interval
                    z = 1.645  # 90% confidence
                    denominator = 1 + z**2/total
                    centre_adjusted_probability = (accuracy + z*z/(2*total))/denominator
                    adjusted_standard_error = z * np.sqrt((accuracy * (1 - accuracy) + z*z/(4*total))/total)/denominator
                    lower_bound = max(0, centre_adjusted_probability - adjusted_standard_error)
                    upper_bound = min(1, centre_adjusted_probability + adjusted_standard_error)
                else:
                    # For edge cases, use simple binomial confidence interval
                    if total > 0:
                        ci = stats.binom.interval(0.90, total, accuracy)
                        lower_bound = ci[0] / total
                        upper_bound = ci[1] / total
                    else:
                        lower_bound = upper_bound = accuracy
                
                bin_stats.append({
                    "conf_bin": bin_name,
                    "conf_mean": conf_mean,
                    "accuracy": accuracy,
                    "count": total,
                    "model_correct": bin_model_correct,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                })
                
                all_confidences.append(conf_mean)
                all_accuracies.append(accuracy)
        
        if not bin_stats:
            continue
            
        bin_df = pd.DataFrame(bin_stats)
        
        # Plot individual points with confidence intervals
        plt.errorbar(bin_df["conf_mean"], bin_df["accuracy"], 
                    yerr=[bin_df["accuracy"] - bin_df["lower_bound"], 
                          bin_df["upper_bound"] - bin_df["accuracy"]],
                    fmt='o', capsize=5, capthick=2, alpha=0.7, 
                    label=f"Bins (n={len(bin_df)})")
        
        # Add linear regression
        if len(all_confidences) >= 2:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            X = np.array(all_confidences).reshape(-1, 1)
            y = np.array(all_accuracies)
            
            # Weight by sample size
            weights = np.array([stats["count"] for stats in bin_stats])
            
            # Fit weighted linear regression
            reg = LinearRegression()
            reg.fit(X, y, sample_weight=weights)
            
            # Predict for plotting
            X_plot = np.linspace(min(all_confidences), max(all_confidences), 100).reshape(-1, 1)
            y_pred = reg.predict(X_plot)
            
            # Calculate confidence interval for regression line
            from scipy import stats as scipy_stats
            
            # Calculate residuals
            y_fit = reg.predict(X)
            residuals = y - y_fit
            n = len(X)  # Number of data points
            mse = np.sum(residuals**2) / (n - 2)
            
            # Calculate prediction intervals
            X_mean = np.mean(X)
            X_var = np.sum((X - X_mean)**2)
            
            # Standard error of prediction
            se_pred = np.sqrt(mse * (1 + 1/n + (X_plot - X_mean)**2 / X_var))
            
            # t-value for 90% confidence
            t_val = scipy_stats.t.ppf(0.95, n - 2)
            
            # Prediction intervals
            y_pred_lower = y_pred - t_val * se_pred.flatten()
            y_pred_upper = y_pred + t_val * se_pred.flatten()
            
            # Plot regression line with confidence interval
            plt.fill_between(X_plot.flatten(), y_pred_lower, y_pred_upper, 
                           alpha=0.3, color='red', label='90% CI')
            plt.plot(X_plot, y_pred, 'r-', linewidth=2, 
                    label=f'Linear fit (R²={r2_score(y, y_fit):.3f})')
        
        # Add threshold line if exists
        if threshold_info["threshold"] is not None:
            plt.axvline(x=threshold_info["threshold"], color='green', linestyle='--', 
                       linewidth=2, label=f"Threshold: {threshold_info['threshold']:.3f}")
        
        plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label="50% accuracy")
        plt.xlabel("Confidence")
        plt.ylabel("Model Accuracy")
        
        # Safe title formatting with overall accuracy
        threshold_str = f"{threshold_info['threshold']:.3f}" if threshold_info["threshold"] is not None else "None"
        method_str = threshold_info["threshold_method"] if threshold_info["threshold_method"] is not None else "None"
        overall_acc = threshold_info["model_accuracy_overall"]
        plt.title(f"{model_name}\nOverall: {overall_acc:.1%} | Threshold: {threshold_str}\n({method_str})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "confidence_thresholds.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.show()
    
    # Summary table
    summary_data = []
    for model_name, threshold_info in thresholds.items():
        summary_data.append({
            "model": model_name,
            "threshold": threshold_info["threshold"],
            "threshold_method": threshold_info["threshold_method"],
            "total_samples": threshold_info["total_samples"],
            "model_correct_count": threshold_info["model_correct_count"],
            "overall_accuracy": threshold_info["model_accuracy_overall"],
            "overall_accuracy_pct": f"{threshold_info['model_accuracy_overall']:.1%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "confidence_thresholds_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM judgments to find confidence thresholds")
    parser.add_argument("--judgments_dir", type=str, default="./artifacts/llm_judgments")
    parser.add_argument("--output_dir", type=str, default="./artifacts/confidence_analysis")
    parser.add_argument("--target_accuracy", type=float, default=0.5, 
                       help="Target accuracy threshold (default: 0.5 = 50%)")
    parser.add_argument("--min_samples_per_bin", type=int, default=5,
                       help="Minimum samples per confidence bin")
    
    args = parser.parse_args()
    
    # Load judgments
    print(f"Loading judgments from {args.judgments_dir}")
    judgments = load_judgments(args.judgments_dir)
    
    if not judgments:
        print("No judgment files found!")
        return
    
    print(f"Found {len(judgments)} models with judgments")
    
    # Calculate thresholds
    print("Calculating confidence thresholds...")
    thresholds = calculate_confidence_thresholds(
        judgments, 
        min_samples_per_bin=args.min_samples_per_bin,
        target_accuracy=args.target_accuracy
    )
    
    # Create plots and summary
    print("Creating analysis plots...")
    summary_df = plot_confidence_analysis(judgments, thresholds, args.output_dir)
    
    # Print results
    print("\n=== Confidence Threshold Analysis ===")
    print(f"Target accuracy: {args.target_accuracy:.1%}")
    print("\nModel Performance Summary:")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']}:")
        print(f"    Overall accuracy in disagreements: {row['overall_accuracy_pct']} ({row['model_correct_count']}/{row['total_samples']})")
        if pd.notna(row["threshold"]):
            print(f"    Confidence threshold: {row['threshold']:.3f} ({row['threshold_method']})")
            print(f"    → Trust model above {row['threshold']:.1%} confidence")
        else:
            print(f"    Confidence threshold: None (never right >{args.target_accuracy:.0%} when disagreeing)")
        print()
    
    print(f"\nDetailed results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
