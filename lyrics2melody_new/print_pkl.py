import pickle

# Read the pickle file
with open("/volume/fundwo-test/lyrics2melody/share_REMI_aligned/不会再让你哭-祁隆-134-Ab.pkl", "rb") as f:
    data = pickle.load(f)

# Check the type of the object
print(type(data))

# Print or explore the content
print(data)
print(len(data[0]))
