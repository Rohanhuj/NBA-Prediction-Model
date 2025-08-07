import pickle

# Load the old pickle file
with open("dict23-24.pkl", "rb") as f:
    data = pickle.load(f)

# Re-save with a modern, portable protocol
with open("team23_24_clean.pkl", "wb") as f:
    pickle.dump(data, f, protocol=4)

print("Re-pickled successfully to 'team23_24_clean.pkl'")
