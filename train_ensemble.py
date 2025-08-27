from __future__ import annotations

import argparse
import pandas as pd
from ensemble import EnsemblePipeline, EnsembleConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/unique_products.csv')
    parser.add_argument('--label_column', type=str, default='Libellé produit')
    parser.add_argument('--target_column', type=str, default='Nature')
    parser.add_argument('--tfidf', action='store_true')
    parser.add_argument('--sent', action='store_true')
    parser.add_argument('--camembert', action='store_true')
    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[[args.label_column, args.target_column]].dropna()
    df = df[(df[args.label_column].astype(str).str.strip() != '') & (df[args.target_column].astype(str).str.strip() != '')]
    X = df[args.label_column].astype(str).tolist()
    y = df[args.target_column].astype(str).tolist()

    cfg = EnsembleConfig(
        use_tfidf=args.tfidf or (not args.sent and not args.camembert and not args.knn),
        use_sent_emb=args.sent,
        use_camembert=args.camembert,
        use_knn=args.knn,
        max_samples=args.max_samples,
    )
    pipeline = EnsemblePipeline(cfg)
    X_train, X_test, y_train, y_test = pipeline.split(X, y)
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_test, y_test)
    print(metrics)


if __name__ == '__main__':
    main()


