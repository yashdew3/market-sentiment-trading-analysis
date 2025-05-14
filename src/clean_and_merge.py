import pandas as pd
import os

def clean_and_merge_data(sentiment_path, trader_path, output_path):
    """Cleans and merges sentiment and trader datasets on date level."""
    print("🔄 Loading datasets...")
    sentiment_df = pd.read_csv(sentiment_path)
    trader_df = pd.read_csv(trader_path)

    print("📅 Parsing date columns...")
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    trader_df['Timestamp IST'] = pd.to_datetime(trader_df['Timestamp IST'], format="%d-%m-%Y %H:%M")
    trader_df['date'] = trader_df['Timestamp IST'].dt.date
    trader_df['date'] = pd.to_datetime(trader_df['date'])

    print("📊 Aggregating trader data by date...")
    aggregated_trader_df = trader_df.groupby('date').agg({
        'Closed PnL': 'sum',
        'Execution Price': 'mean',
        'Size USD': 'sum',
        'Fee': 'sum',
        'Order ID': 'count'
    }).reset_index()

    aggregated_trader_df.rename(columns={
        'Closed PnL': 'Total_PnL',
        'Execution Price': 'Avg_Execution_Price',
        'Size USD': 'Total_Trade_Volume',
        'Fee': 'Total_Fee',
        'Order ID': 'Trade_Count'
    }, inplace=True)

    print("🔗 Merging with sentiment data...")
    merged_df = pd.merge(
        aggregated_trader_df,
        sentiment_df[['date', 'value', 'classification']],
        on='date',
        how='left'
    )

    print(f"💾 Saving merged data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print("✅ Data processing complete.")

if __name__ == "__main__":
    clean_and_merge_data(
        sentiment_path="data/fear_greed_index.csv",
        trader_path="data/historical_data.csv",
        output_path="data/merged_trader_sentiment_data.csv"
    )
