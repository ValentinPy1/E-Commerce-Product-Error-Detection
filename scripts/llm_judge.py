from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class JudgeConfig:
    provider: str  # 'openai'
    model: str     # e.g., 'gpt-4o-mini'
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_retries: int = 5
    rate_limit_sleep: float = 0.0  # seconds between calls


class OpenAIJudge:
    def __init__(self, cfg: JudgeConfig) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "openai package is required. pip install openai>=1.40") from e
        # Try to load from .env file first, then fallback to environment variable
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {cfg.api_key_env} in .env file "
                f"or environment variable"
            )
        self.client = OpenAI(api_key=api_key)
        self.cfg = cfg

    def _build_messages(
        self,
        label_text: str,
        official_nature: str,
        model_name: str,
        model_pred: str,
    ) -> List[Dict[str, str]]:
        # Prompt in French, dataset is in French
        system = (
            "Tu es un expert en classification de produits français.\n"
            "Je te donne le libellé d'un produit (texte brut), "
            "la catégorie officielle 'Nature' et la catégorie prédite par un modèle.\n"
            "Ta tâche est de décider laquelle est la plus plausible: "
            "la 'Nature' officielle ou la prédiction du modèle.\n"
            "Si aucune des deux n'est plausible, indique que les deux sont fausses.\n"
            "Réponds STRICTEMENT en JSON avec les clés: "
            "verdict (model|nature|both_wrong) et explanation "
            "(une courte justification).\n"
        )
        user = (
            f"Libellé produit: {label_text}\n"
            f"Nature officielle: {official_nature}\n"
            f"Prédiction du modèle ({model_name}): {model_pred}\n"
            "Donne le verdict et une brève justification."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=1, max=30),
        reraise=True,
    )
    def judge_one(
        self,
        label_text: str,
        official_nature: str,
        model_name: str,
        model_pred: str,
    ) -> Dict[str, str]:
        messages = self._build_messages(
            label_text, official_nature, model_name, model_pred)
        # Some models don't support temperature=0.0, so we omit it for those
        kwargs = {
            "model": self.cfg.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        # Only add temperature if it's not 0.0 (which some models don't support)
        if self.cfg.temperature != 0.0:
            kwargs["temperature"] = self.cfg.temperature
        resp = self.client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except Exception:
            data = {"verdict": "error", "explanation": raw}
        return data


def build_judge(cfg: JudgeConfig):
    if cfg.provider == "openai":
        return OpenAIJudge(cfg)
    raise RuntimeError(f"Unsupported provider: {cfg.provider}")


def collect_models(pred_df: pd.DataFrame) -> List[str]:
    names = sorted({c[:-5] for c in pred_df.columns if c.endswith("_pred")})
    # exclude 'Nature' if it accidentally matches suffix logic
    names = [n for n in names if n not in ("Nature",)]
    return names


def run():
    parser = argparse.ArgumentParser(
        description="LLM-assisted judging of model vs Nature on disagreements"
    )
    parser.add_argument("--predictions", type=str,
                        default="./artifacts/ensemble_predictions.csv")
    parser.add_argument("--label_column", type=str, default="Libellé produit")
    parser.add_argument("--target_column", type=str, default="Nature")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-5-mini-2025-08-07")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument(
        "--samples_per_model",
        type=int,
        default=100,
        help="Number of samples to judge per model",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str,
                        default="./artifacts/llm_judgments")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if args.label_column not in df.columns or args.target_column not in df.columns:
        raise SystemExit("Missing label/target columns in predictions CSV")

    models = collect_models(df)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = JudgeConfig(
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env,
        rate_limit_sleep=args.sleep,
    )
    judge = build_judge(cfg)

    summary_rows: List[Dict[str, object]] = []

    for model_name in models:
        pred_col = f"{model_name}_pred"
        conf_col = f"{model_name}_conf"
        if pred_col not in df.columns:
            continue

        # Get disagreements only
        mask = df[pred_col].notna() & (
            df[pred_col].astype(str) != df[args.target_column].astype(str)
        )
        sub = df.loc[mask, [args.label_column,
                            args.target_column, pred_col]].copy()

        # Add confidence column if available
        if conf_col in df.columns:
            sub[conf_col] = df.loc[mask, conf_col]

        if sub.empty:
            continue

        # Sample evenly across confidence ranges
        if conf_col in sub.columns and len(sub) > args.samples_per_model:
            # Create confidence bins and sample evenly from each
            sub["conf_bin"] = pd.cut(
                sub[conf_col], bins=10, labels=False, include_lowest=True
            )
            sampled_indices = []

            for bin_idx in range(10):
                bin_mask = sub['conf_bin'] == bin_idx
                bin_size = bin_mask.sum()
                if bin_size > 0:
                    # Sample proportionally from this bin
                    samples_from_bin = max(
                        1, int(args.samples_per_model * bin_size / len(sub))
                    )
                    bin_indices = sub[bin_mask].index.tolist()
                    sampled_from_bin = np.random.choice(
                        bin_indices,
                        size=min(samples_from_bin, len(bin_indices)),
                        replace=False,
                    )
                    sampled_indices.extend(sampled_from_bin)

            # If we didn't get enough samples, add random ones
            if len(sampled_indices) < args.samples_per_model:
                remaining_indices = [
                    idx for idx in sub.index if idx not in sampled_indices]
                additional_needed = args.samples_per_model - \
                    len(sampled_indices)
                if remaining_indices:
                    additional_indices = np.random.choice(
                        remaining_indices,
                        size=min(additional_needed, len(remaining_indices)),
                        replace=False,
                    )
                    sampled_indices.extend(additional_indices)

            # Take exactly the requested number
            sampled_indices = sampled_indices[:args.samples_per_model]
            sub = sub.loc[sampled_indices]
        else:
            # If no confidence or not enough data, just take the first N
            sub = sub.head(args.samples_per_model)

        out_path = os.path.join(args.out_dir, f"{model_name}_judgments.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            cols = [
                "label",
                "official_nature",
                "model",
                "model_pred",
                "model_conf",
                "verdict",
                "explanation",
            ]
            writer.writerow(cols)

            correct_model = 0
            correct_nature = 0
            both_wrong = 0
            errors = 0

            for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"Judging {model_name}"):
                label_text = str(row[args.label_column])
                official = str(row[args.target_column])
                model_pred = str(row[pred_col])
                model_conf = (
                    float(row.get(conf_col, np.nan)
                          ) if conf_col in row else np.nan
                )

                try:
                    result = judge.judge_one(
                        label_text, official, model_name, model_pred)
                except Exception as e:
                    result = {"verdict": "error", "explanation": str(e)}
                verdict = result.get("verdict", "error")
                expl = result.get("explanation", "")
                writer.writerow(
                    [
                        label_text,
                        official,
                        model_name,
                        model_pred,
                        model_conf,
                        verdict,
                        expl,
                    ]
                )

                if verdict == "model":
                    correct_model += 1
                elif verdict == "nature":
                    correct_nature += 1
                elif verdict == "both_wrong":
                    both_wrong += 1
                else:
                    errors += 1

                if cfg.rate_limit_sleep > 0:
                    time.sleep(cfg.rate_limit_sleep)

        # Add trivial equals (where model==nature) as both-correct
        equal_count = int(
            (df[pred_col].astype(str) == df[args.target_column].astype(str)).sum()
        )
        total_judged = len(sub)
        total_disagreements = int(mask.sum())

        summary_rows.append({
            "model": model_name,
            "equal_both_correct": equal_count,
            "total_disagreements": total_disagreements,
            "judged_sample_size": total_judged,
            "verdict_model": correct_model,
            "verdict_nature": correct_nature,
            "verdict_both_wrong": both_wrong,
            "verdict_error": errors,
            "model_accuracy_in_disagreements": (
                correct_model / total_judged if total_judged > 0 else 0
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved summary to", summary_path)


if __name__ == "__main__":
    run()
