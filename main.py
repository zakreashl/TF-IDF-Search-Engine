from datasets import load_dataset
import pandas

dataset = load_dataset("NeelNanda/wiki-10k")

df = dataset["train"].to_pandas()

print(df.head())