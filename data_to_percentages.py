import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('data.csv')

# Drop the 'Date' and 'Symbol' columns
df = df.drop(columns=['Date', 'Symbol'])

# Reverse the order of the rows
df = df.iloc[::-1]

# Convert all columns to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Calculate percentage changes
df_pct_change = df.pct_change().fillna(0) * 100

# Reverse the order back to original
df_pct_change = df_pct_change.iloc[::-1]

# Save the results to a new CSV file
df_pct_change.to_csv('percentage_changes.csv', index=False)

print("Percentage changes have been saved to 'percentage_changes.csv'.")