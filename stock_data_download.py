import pandas_datareader.data as web
import datetime
import os

# Define start and end dates
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2025, 1, 1)

# Fetch AAPL data from Stooq (since Google Finance is not available in pandas_datareader)
df = web.DataReader("AAPL", "stooq", start, end)

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
