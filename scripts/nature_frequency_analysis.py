#!/usr/bin/env python3
"""
Nature Label Frequency vs Error Rate Analysis

This script analyzes how the frequency of nature labels across the dataset 
affects the error rate for each model. It provides comprehensive insights
into model performance across different frequency bins and individual labels.
"""

import os
import warnings
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend
warnings.filterwarnings("ignore")


def main():
    print("=== Nature Label Frequency vs Error Rate Analysis ===")

    # Load ensemble predictions
    preds_path = 'artifacts/results/ensemble_predictions.csv'
    if not os.path.exists(preds_path):
        print(f"Error: Predictions file not found at {preds_path}")
        return

    df = pd.read_csv(preds_path)

    # Get model names from prediction columns
    pred_cols = [c for c in df.columns if c.endswith('_pred')]
    model_names = sorted({c[:-5] for c in pred_cols})

    print(f"Models found: {model_names}")
    print(f"Total samples: {len(df)}")

    # Calculate nature label frequencies
    nature_counts = df['Nature'].value_counts()
    print(f"\nNature labels found: {len(nature_counts)}")
    print(f"Most common nature labels:")
    print(nature_counts.head(10))

    # Function to calculate error rate for each model per nature label frequency
    def calculate_error_rates_by_nature(df, model_names):
        """Calculate error rate for each model grouped by nature label frequency"""

        # Create frequency bins for nature labels
        nature_counts = df['Nature'].value_counts()

        # Define frequency bins
        freq_bins = [
            (1, 10, 'Very Rare (1-10)'),
            (11, 50, 'Rare (11-50)'),
            (51, 200, 'Uncommon (51-200)'),
            (201, 1000, 'Common (201-1000)'),
            (1001, float('inf'), 'Very Common (1000+)')
        ]

        results = []

        for min_freq, max_freq, bin_label in freq_bins:
            # Get nature labels in this frequency bin
            if max_freq == float('inf'):
                mask = nature_counts >= min_freq
            else:
                mask = (nature_counts >= min_freq) & (
                    nature_counts <= max_freq)

            bin_natures = nature_counts[mask].index

            if len(bin_natures) == 0:
                continue

            # Filter data for this frequency bin
            bin_data = df[df['Nature'].isin(bin_natures)]

            if len(bin_data) == 0:
                continue

            # Calculate error rate for each model
            for model in model_names:
                pred_col = f'{model}_pred'
                if pred_col in df.columns:
                    # Calculate errors
                    errors = (bin_data['Nature'] != bin_data[pred_col]).sum()
                    total = len(bin_data)
                    error_rate = errors / total if total > 0 else 0

                    results.append({
                        'frequency_bin': bin_label,
                        'model': model,
                        'total_samples': total,
                        'errors': errors,
                        'error_rate': error_rate,
                        'avg_frequency': nature_counts[bin_natures].mean()
                    })

        return pd.DataFrame(results)

    # Calculate error rates by frequency
    error_analysis = calculate_error_rates_by_nature(df, model_names)

    print("\n=== Error Rate Analysis by Nature Label Frequency ===")
    print(error_analysis)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Error Rate by Frequency Bin for Each Model
    plt.subplot(2, 2, 1)
    pivot_data = error_analysis.pivot(
        index="frequency_bin", columns="model", values="error_rate"
    )

    # Define correct order for frequency bins
    freq_order = [
        'Very Rare (1-10)',
        'Rare (11-50)',
        'Uncommon (51-200)',
        'Common (201-1000)',
        'Very Common (1000+)'
    ]

    # Reorder the pivot data to match the logical sequence
    pivot_data = pivot_data.reindex(freq_order)

    pivot_data.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Error Rate by Nature Label Frequency Bin')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Error Rate')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 2: Sample Count by Frequency Bin
    plt.subplot(2, 2, 2)
    sample_counts = error_analysis.groupby("frequency_bin")[
        "total_samples"
    ].first()

    # Reorder sample counts to match frequency bin order
    sample_counts = sample_counts.reindex(freq_order)

    sample_counts.plot(kind='bar', ax=plt.gca(), color='skyblue', width=0.8)
    plt.title('Sample Count by Frequency Bin')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 3: Error Rate vs Average Frequency (scatter plot)
    plt.subplot(2, 2, 3)
    for model in model_names:
        model_data = error_analysis[error_analysis["model"] == model]
        plt.scatter(
            model_data["avg_frequency"],
            model_data["error_rate"],
            label=model,
            alpha=0.7,
            s=100,
        )
    plt.xlabel('Average Frequency of Nature Labels in Bin')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Nature Label Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')

    # Plot 4: Heatmap of Error Rates
    plt.subplot(2, 2, 4)
    heatmap_data = error_analysis.pivot(
        index="frequency_bin", columns="model", values="error_rate"
    )

    # Reorder heatmap to match frequency bin order
    heatmap_data = heatmap_data.reindex(freq_order)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        cbar_kws={"label": "Error Rate"},
    )
    plt.title('Error Rate Heatmap by Model and Frequency')

    plt.tight_layout()
    print("Saving first visualization...")
    plt.savefig(
        "artifacts/nature_frequency_error_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("First visualization saved successfully")
    plt.close()  # Close the figure to free memory

    # Detailed analysis by individual nature labels
    print("\n=== Detailed Analysis by Individual Nature Labels ===")

    # Get top 20 most frequent nature labels
    top_natures = nature_counts.head(20).index

    detailed_results = []
    for nature in top_natures:
        nature_data = df[df['Nature'] == nature]

        for model in model_names:
            pred_col = f'{model}_pred'
            if pred_col in df.columns:
                errors = (nature_data['Nature'] != nature_data[pred_col]).sum()
                total = len(nature_data)
                error_rate = errors / total if total > 0 else 0

                detailed_results.append({
                    'nature': nature,
                    'model': model,
                    'frequency': nature_counts[nature],
                    'total_samples': total,
                    'errors': errors,
                    'error_rate': error_rate
                })

    detailed_df = pd.DataFrame(detailed_results)

    # Create detailed visualization
    plt.figure(figsize=(20, 12))

    # Plot 1: Error Rate by Nature Label (top 20)
    plt.subplot(2, 1, 1)
    pivot_detailed = detailed_df.pivot(
        index="nature", columns="model", values="error_rate"
    )

    # Sort nature labels by frequency (most frequent first)
    nature_freq_order = (
        detailed_df.groupby("nature")["frequency"]
        .first()
        .sort_values(ascending=False)
        .index
    )
    pivot_detailed = pivot_detailed.reindex(nature_freq_order)

    pivot_detailed.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('Error Rate by Nature Label (Top 20 Most Frequent)')
    plt.xlabel('Nature Label')
    plt.ylabel('Error Rate')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 2: Frequency vs Error Rate scatter for each model
    plt.subplot(2, 1, 2)
    for model in model_names:
        model_data = detailed_df[detailed_df["model"] == model]
        plt.scatter(
            model_data["frequency"],
            model_data["error_rate"],
            label=model,
            alpha=0.7,
            s=100,
        )
    plt.xlabel('Frequency of Nature Label')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Nature Label Frequency (Individual Labels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')

    plt.tight_layout()
    print("Saving second visualization...")
    plt.savefig(
        "artifacts/detailed_nature_error_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Second visualization saved successfully")
    plt.close()  # Close the figure to free memory

    # Statistical analysis
    print("\n=== Statistical Analysis ===")

    # Correlation between frequency and error rate for each model
    print("Correlation between Nature Label Frequency and Error Rate:")
    for model in model_names:
        model_data = detailed_df[detailed_df['model'] == model]
        if len(model_data) > 1:
            correlation = model_data['frequency'].corr(
                model_data['error_rate'])
            print(f"{model}: {correlation:.4f}")

    # Summary statistics by frequency bin
    print("\nSummary Statistics by Frequency Bin:")
    summary_stats = (
        error_analysis.groupby("frequency_bin")
        .agg(
            {
                "error_rate": ["mean", "std", "min", "max"],
                "total_samples": "sum",
            }
        )
        .round(4)
    )

    # Reorder summary stats to match frequency bin order
    summary_stats = summary_stats.reindex(freq_order)
    print(summary_stats)

    # Save detailed results
    os.makedirs('artifacts', exist_ok=True)
    detailed_df.to_csv(
        'artifacts/nature_frequency_detailed_analysis.csv', index=False)
    error_analysis.to_csv(
        'artifacts/nature_frequency_error_analysis.csv', index=False)

    print("\n=== Analysis Complete ===")
    print("Files saved:")
    print("- nature_frequency_error_analysis.png")
    print("- detailed_nature_error_analysis.png")
    print("- nature_frequency_error_analysis.csv")
    print("- nature_frequency_detailed_analysis.csv")

    # Additional insights
    print("\n=== Key Insights ===")

    # Find best and worst performing models by frequency bin
    for bin_name in freq_order:
        bin_data = error_analysis[error_analysis['frequency_bin'] == bin_name]
        if len(bin_data) > 0:
            best_model = bin_data.loc[bin_data['error_rate'].idxmin()]
            worst_model = bin_data.loc[bin_data['error_rate'].idxmax()]
            print(f"\n{bin_name}:")
            print(
                f"  Best: {best_model['model']} (Error Rate: {best_model['error_rate']:.3f})")
            print(
                f"  Worst: {worst_model['model']} (Error Rate: {worst_model['error_rate']:.3f})")

    # Find nature labels with highest error rates across all models
    print("\nNature Labels with Highest Error Rates (Top 10):")
    avg_errors = detailed_df.groupby(
        'nature')['error_rate'].mean().sort_values(ascending=False)
    print(avg_errors.head(10))

    # Find nature labels with lowest error rates across all models
    print("\nNature Labels with Lowest Error Rates (Top 10):")
    print(avg_errors.tail(10))

    # Show nature labels ordered by frequency
    print("\nNature Labels Ordered by Frequency (Top 20):")
    freq_ordered = detailed_df.groupby(
        'nature')['frequency'].first().sort_values(ascending=False)
    print(freq_ordered.head(20))


if __name__ == "__main__":
    main()
