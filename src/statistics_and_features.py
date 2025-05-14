import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr
import os

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def encode_sentiment(df):
    # Binary encoding: 1 for Greed, 0 for Fear
    df['Sentiment_Binary'] = df['classification'].map({'Greed': 1, 'Fear': 0})
    return df

def add_rolling_features(df):
    # Sort for proper rolling
    df = df.sort_values('date')

    # Rolling metrics (3-day window)
    df['PnL_3D_Mean'] = df['Total_PnL'].rolling(window=3).mean()
    df['PnL_3D_STD'] = df['Total_PnL'].rolling(window=3).std()

    df['Volume_3D_Mean'] = df['Total_Trade_Volume'].rolling(window=3).mean()
    df['TradeCount_3D_Mean'] = df['Trade_Count'].rolling(window=3).mean()

    return df

def perform_statistical_tests(df):
    # Filter based on sentiment
    fear_df = df[df['classification'] == 'Fear']
    greed_df = df[df['classification'] == 'Greed']

    tests = {}

    # Apply t-tests / Mann-Whitney for Total PnL
    for col in ['Total_PnL', 'Total_Trade_Volume', 'Trade_Count']:
        stat, pval = ttest_ind(fear_df[col], greed_df[col], equal_var=False)
        tests[col] = {
            "test": "t-test",
            "statistic": stat,
            "p-value": pval
        }

    # Spearman correlation with numeric sentiment
    correlations = {}
    for col in ['Total_PnL', 'Total_Trade_Volume', 'Trade_Count']:
        corr, pval = spearmanr(df['value'], df[col])
        correlations[col] = {
            "correlation": corr,
            "p-value": pval
        }

    return tests, correlations

def save_transformed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature-enriched data saved to: {output_path}")

def save_results(stat_tests, correlations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(stat_tests).T.to_csv(os.path.join(output_dir, "statistical_tests.csv"))
    pd.DataFrame(correlations).T.to_csv(os.path.join(output_dir, "correlation_results.csv"))
    print(f"📄 Test results saved in: {output_dir}")

if __name__ == "__main__":
    input_path = "data/merged_trader_sentiment_data.csv"
    output_data_path = "data/feature_engineered_data.csv"
    output_results_path = "outputs/statistics/"

    df = load_data(input_path)
    df = encode_sentiment(df)
    df = add_rolling_features(df)
    
    stat_tests, correlations = perform_statistical_tests(df)
    save_transformed_data(df, output_data_path)
    save_results(stat_tests, correlations, output_results_path)
