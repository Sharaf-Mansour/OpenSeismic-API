import pandas as pd

def get_trace(start_date, end_date):
    # Load the CSV file
    df = pd.read_csv('Waveforms/metadata.csv')

    # Convert the date column to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Filter the dataframe by the date range
    mask = (df['time'] >= start_date) & (df['time'] <= end_date)
    filtered_df = df.loc[mask]

    # Print the index of each row that fits in the date range
    return filtered_df.index.tolist()

 