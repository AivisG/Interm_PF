import pandas_datareader.data as web
import datetime
import os

# Define start and end dates
start = datetime.datetime(2022, 1, 1)
end = datetime.datetime(2025, 1, 1)

# Fetch AAPL data from Stooq (Stooq provides data in reverse order)
df = web.DataReader("AAPL", "stooq", start, end)

# Sort the data in ascending order to have oldest dates first
df = df.sort_index(ascending=True)  # ✅ Fix: Sort by date in ascending order

# Format dates for filename (YYYY-MM-DD format)
start_str = start.strftime("%Y-%m-%d")
end_str = end.strftime("%Y-%m-%d")

# Create directory if not exists
directory = "stock_data_csv"
os.makedirs(directory, exist_ok=True)

# Define the file path with start and end date
file_path = os.path.join(directory, f"AAPL_{start_str}_to_{end_str}.csv")

# Save as CSV
df.to_csv(file_path)
print(f"Saved {file_path} successfully!")

# Print first few rows to verify the fix
print(df.head(10))  # Oldest data should appear first
