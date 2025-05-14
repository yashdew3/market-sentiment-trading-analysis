import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def run_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Summary stats
    print("📊 Basic Summary:")
    print(df.describe())

    # Check missing values
    print("\n🧼 Missing Values:")
    print(df.isnull().sum())

    # Distribution of Sentiment
    sentiment_counts = df['classification'].value_counts()
    print("\n🔁 Sentiment Distribution:")
    print(sentiment_counts)

    # Save sentiment barplot
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='classification', palette='coolwarm')
    plt.title("Distribution of Sentiment (Fear/Greed)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"))
    plt.close()

    # Line plots: PnL, Volume, and Trade Count over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='date', y='Total_PnL', hue='classification')
    plt.title("Total PnL Over Time by Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pnl_over_time.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='date', y='Total_Trade_Volume', hue='classification')
    plt.title("Total Trade Volume Over Time by Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "volume_over_time.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='date', y='Trade_Count', hue='classification')
    plt.title("Trade Count Over Time by Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trade_count_over_time.png"))
    plt.close()

    print(f"\n✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    data_path = "data/merged_trader_sentiment_data.csv"
    output_path = "outputs/eda/"
    
    df = load_data(data_path)
    run_eda(df, output_path)
