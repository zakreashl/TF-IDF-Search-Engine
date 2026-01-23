from datasets import load_dataset, DataFrame

# Many commonly used words can interfer with search prompt so this is a list of those words which we will use to remove from text later
STOP_WORDS: set = ["a", "to", "the", "is", "in", "and", "was", "with", "his", "her", "that", "for", "from", "such", 
                   "which", "their", "this", "were", "has", "are", "its", "have", "not", "they", "also", "during",
                   "many", "would", "been", "some", "but", "other", "among", "being", "had", "more", "most", "can",
                   "into", "who", "than", "while", "first", "means", "first", "both", "all", "became", "because"]

dataset = load_dataset("NeelNanda/wiki-10k")
df = dataset["train"].to_pandas() # Convert the dataset to a pandas for easier use

def search(search_prompt: str, df: DataFrame) -> int:
    index: int = 0
    greatest_frequency: float = 0

    for i in range(len(df)):
        term_frequencies = df["term_frequencies"].iloc[i]

        if search_prompt not in term_frequencies:
            continue

        if term_frequencies.get(search_prompt) > greatest_frequency:
            index = i
            greatest_frequency = term_frequencies[search_prompt]

    return index

def process_clean_text(text: str, stop_words: set) -> list:
    processed = []

    # Simplify the text
    text = text.lower() # Set all letters to lowercase for simplicity
    text = text.replace('[\\(\\);\\:\\-\\–,.!?\n]', '', regex=True) # Remove punctuation
    text = text.strip() # Remove trailing white space
    text = text.split(" ") # Turn into list of words

    # Remove all stop words
    for word in text:
        if word not in stop_words and len(word) > 2:
            processed.append(word)

    return processed

def generate_term_frequencies(clean_text: list) -> dict:
    term_frequencies: dict = {}

    # Get all the counts of each meaningful word in the document
    for word in clean_text:
        term_frequencies[word] = term_frequencies.get(word, 0) + 1.0

    # Now normalize all the term frequencies so they are easier to work with
    for key in term_frequencies.keys():
        term_frequencies[key] = term_frequencies[key] / len(term_frequencies)

    return term_frequencies


df["clean_text"] = (
    df["text"]
    .apply(process_clean_text, stop_words=STOP_WORDS) # Process the text in a custom function for more complex proccesing 
)

df["term_frequencies"] = (
    df["clean_text"]
    .apply(generate_term_frequencies) # Generate the term frequencies with a custom function for more complex proccesing 
)

print("Done processing text")

print(df.columns)
#print(df["term_frequencies"].iloc[0])


prompt = str(input("Search: "))

while True:
    prompt = str(input("Search: "))
    if prompt == "done": break
    article_index = search(prompt, df=df)
    print(df["title"].iloc[article_index])
    print(df["text"].iloc[article_index])