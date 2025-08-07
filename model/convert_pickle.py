import pickle
import pandas as pd

# Load the legacy pickle file
with open("dict23-24.pkl", "rb") as f:
    data = pickle.load(f)

# Helper function to strip problematic pandas index types
def strip_indexes(obj):
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.index = range(len(df))  # Replace index with plain RangeIndex
        df.columns = list(df.columns)  # Ensure columns are plain list
        return df
    elif isinstance(obj, pd.Series):
        s = obj.copy()
        s.index = range(len(s))
        return s
    elif isinstance(obj, dict):
        return {k: strip_indexes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_indexes(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(strip_indexes(i) for i in obj)
    else:
        return obj

# Apply the cleanup
clean_data = strip_indexes(data)

# Save to a new, safe pickle file
with open("team23_24_index_cleaned.pkl", "wb") as f:
    pickle.dump(clean_data, f, protocol=4)

print("✅ Cleaned and saved as 'team23_24_index_cleaned.pkl'")
