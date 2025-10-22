import pandas as pd

csv_path = "annotations/synthetic_annotations_noheader.csv"
df = pd.read_csv(csv_path, header=None)

# Replace Windows slashes and remove leading "sentences_data_synth\" or "sentences_data_synth/"
df[0] = df[0].str.replace("\\", "/", regex=False)
df[0] = df[0].str.replace(r"^sentences_data_synth/", "", regex=True)

df.to_csv(csv_path, header=False, index=False)
print("✅ Fixed annotation paths and saved to", csv_path)