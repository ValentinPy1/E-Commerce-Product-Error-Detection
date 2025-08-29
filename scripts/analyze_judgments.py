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
            
        # Create confidence bins
        valid["conf_bin"] = pd.cut(valid["model_conf"], bins=10, include_lowest=True)
        
        # Calculate accuracy per bin
        bin_stats = []
        for bin_name, bin_data in valid.groupby("conf_bin", observed=False):
            if len(bin_data) >= min_samples_per_bin:
                model_correct = (bin_data["verdict"] == "model").sum()
                total = len(bin_data)
                accuracy = model_correct / total
                conf_mean = bin_data["model_conf"].mean()
                bin_stats.append({
                    "conf_bin": bin_name,
                    "conf_mean": conf_mean,
                    "accuracy": accuracy,
                    "count": total,
                    "model_correct": model_correct
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
        
        # Method 1: Linear interpolation between bins
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
        
        # Method 2: Logistic regression if linear interpolation fails
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
        
        # Method 3: Simple threshold at first bin above target accuracy
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
            "model_accuracy_overall": (valid["verdict"] == "model").mean()
        }
    
    return results


def plot_confidence_analysis(judgments: Dict[str, pd.DataFrame], 
                           thresholds: Dict[str, Dict],
                           output_dir: str):
    """Create plots for confidence analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall accuracy vs confidence plot
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, df) in enumerate(judgments.items()):
        if model_name not in thresholds:
            continue
            
        plt.subplot(2, 3, i + 1)
        
        # Filter valid judgments
        valid = df[df["verdict"].isin(["model", "nature"])].copy()
        if len(valid) < 10:
            continue
        
        # Create confidence bins
        valid["conf_bin"] = pd.cut(valid["model_conf"], bins=10, include_lowest=True)
        
        # Calculate accuracy per bin
        bin_stats = []
        for bin_name, bin_data in valid.groupby("conf_bin"):
            if len(bin_data) >= 3:
                model_correct = (bin_data["verdict"] == "model").sum()
                total = len(bin_data)
                accuracy = model_correct / total
                conf_mean = bin_data["model_conf"].mean()
                bin_stats.append({
                    "conf_mean": conf_mean,
                    "accuracy": accuracy,
                    "count": total
                })
        
        if not bin_stats:
            continue
            
        bin_df = pd.DataFrame(bin_stats)
        
        # Plot
        plt.scatter(bin_df["conf_mean"], bin_df["accuracy"], 
                   s=bin_df["count"] * 2, alpha=0.7, label=f"Bins (n={len(bin_df)})")
        
        # Add threshold line
        threshold_info = thresholds[model_name]
        if threshold_info["threshold"] is not None:
            plt.axvline(x=threshold_info["threshold"], color='red', linestyle='--', 
                       label=f"Threshold: {threshold_info['threshold']:.3f}")
        
        plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label="50% accuracy")
        plt.xlabel("Confidence")
        plt.ylabel("Model Accuracy")
        plt.title(f"{model_name}\nThreshold: {threshold_info['threshold']:.3f} ({threshold_info['threshold_method']})")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_thresholds.png"), dpi=200, bbox_inches='tight')
    plt.show()
    
    # Summary table
    summary_data = []
    for model_name, threshold_info in thresholds.items():
        summary_data.append({
            "model": model_name,
            "threshold": threshold_info["threshold"],
            "threshold_method": threshold_info["threshold_method"],
            "total_samples": threshold_info["total_samples"],
            "overall_accuracy": threshold_info["model_accuracy_overall"]
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
    print("\nRecommended thresholds (trust model above this confidence):")
    for _, row in summary_df.iterrows():
        if pd.notna(row["threshold"]):
            print(f"  {row['model']}: {row['threshold']:.3f} ({row['threshold_method']})")
        else:
            print(f"  {row['model']}: Could not determine threshold")
    
    print(f"\nDetailed results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
