from datasets import load_dataset
from pandas import DataFrame
from math import log
import re

# Many commonly used words can interfer with search prompt so this is a list of those words which we will use to remove from text later
STOP_WORDS: set[str] = {"a", "to", "the", "is", "in", "and", "was", "with", "his", "her", "that", "for", "from", "such", 
                   "which", "their", "this", "were", "has", "are", "its", "have", "not", "they", "also", "during",
                   "many", "would", "been", "some", "but", "other", "among", "being", "had", "more", "most", "can",
                   "into", "who", "than", "while", "means", "first", "both", "all", "became", "because"}

def search(query: str, articles: DataFrame, idfs: dict[str, float], stop_words: set[str]) -> int:
    query_terms = process_clean_text(query, stop_words)

    # Get unique query term counts  
    query_tf = {}
    for w in query_terms:
        query_tf[w] = query_tf.get(w, 0) + 1

    # Normalize the terms
    for w in query_tf:
        query_tf[w] /= len(query_terms)

    # Make a dict containg all the tf-idfs for easier access
    query_tfidf = {
        w: query_tf[w] * idfs.get(w, 0.0)
        for w in query_tf
    }

    best_idx = -1
    best_score = 0.0

    # Go for each article
    for i in range(len(articles)):
        doc_tf = articles["term_frequencies"].iloc[i]

        # Get the id-idfs and fine the greatest one
        score = 0.0
        for w, q_w in query_tfidf.items():
            if w in doc_tf:
                score += q_w * (doc_tf[w] * idfs[w])

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx

def process_clean_text(text: str, stop_words: set[str]) -> list[str]:
    processed = []

    # Simplify the text
    text = text.lower() # Set all letters to lowercase for simplicity
    text = re.sub(r"[()\[\];:\-–,.!?\n]", "", text) # Remove punctuation
    text = text.strip() # Remove trailing white space
    text = text.split(" ") # Turn into list of words

    # Remove all stop words
    for word in text:
        if word not in stop_words and len(word) > 2:
            processed.append(word)

    return processed

def generate_term_frequencies(clean_text: list[str], idfs: dict[str, float]) -> dict[str, float]:
    term_frequencies: dict = {}

    for word in set(clean_text):
        idfs[word] = idfs.get(word, 0) + 1.0

    # Get all the counts of each meaningful words in the document
    for word in clean_text:
        term_frequencies[word] = term_frequencies.get(word, 0) + 1.0

    # Now normalize all the term frequencies so they are easier to work with
    total_words = sum(term_frequencies.values())
    for key in term_frequencies.keys():
        term_frequencies[key] = term_frequencies[key] / total_words

    return term_frequencies

def generate_article_info(articles: DataFrame, stop_words: set[str], idfs: dict[str, float]):
    articles["clean_text"] = (
        articles["text"]
        .apply(process_clean_text, stop_words=stop_words) # Process the text in a custom function for more complex proccesing 
    )

    articles["term_frequencies"] = (
        articles["clean_text"]
        .apply(generate_term_frequencies, idfs=idfs) # Generate the term frequencies with a custom function for more complex proccesing 
    )

def load_data():
    dataset = load_dataset("NeelNanda/wiki-10k")
    return dataset["train"].to_pandas() # Convert the dataset to a pandas for easier use

def calculate_idfs(idfs: dict[str, float], articles: DataFrame) -> dict[str, float]:
    article_count = len(articles)

    for word in idfs:
        idfs[word] = log(article_count/idfs[word])
    
    return idfs

def main(stop_words: set):
    # Load the data
    articles = load_data()

    idfs: dict[str, float] = {}

    # Process the articles
    print("Processing articles...")
    generate_article_info(articles=articles, stop_words=stop_words, idfs=idfs)
    idfs = calculate_idfs(idfs=idfs, articles=articles)
    print("Done processing articles")

    while True:
        prompt = str(input("Search: ")).lower()
        if prompt == "done": exit()
        article_index = search(prompt, articles=articles, idfs=idfs, stop_words=STOP_WORDS)

        if article_index == -1:
            print("Article not found")
        else:
            print("=======", articles["title"].iloc[article_index], "=======")
            print("\n")
            print(articles["text"].iloc[article_index], "\n\n")

if __name__ == "__main__":
    main(stop_words=STOP_WORDS)