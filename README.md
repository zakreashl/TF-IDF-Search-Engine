# TF-IDF-Search-Engine
A simple TF-IDF based search engine that can process multi-lined prompts and return a resonable article from a list of 10,000 wikipedia articles

This project was built to understand how traditional information retrieval systems work

Featrues:
    - Text proprocessing (lowercasing, puncuation removal, stop-word filtering)
    - Term Frequency (TF) computation per document
    - Inverse Document Frequency (IDF) computation over the articles
    - TF-IDF–based ranking for single- and multi-word queries
    - Interactive command-line search interface


How it works:
    1. 10,000 wikipedia are imported and preprocessed (remove puncuation and uneeded words)
    2. Each article is searched to find its unique terms and how prominent they are in the article which is the TF (term frequency)
    3. Next the IDF (inverse document frequency) is calculated to know how rare every unique word is across all articles
    4. Get the user's search
        - The search prompt is preprocessed by removing uneeded words and lowercased
        - A TF-IDF vector is built for the query
        - Each document is scored using a dot product between TF-IDF vectors
    5. The article with the highest scoring TF-IDF is returned

<img width="778" height="713" alt="Image" src="https://github.com/user-attachments/assets/995c6aaf-d22b-4d1d-8b0c-38c8578ee1dd" />

<img width="778" height="405" alt="Image" src="https://github.com/user-attachments/assets/f8287d35-4871-485e-b042-58fb8658c1d5" />

<img width="778" height="691" alt="Image" src="https://github.com/user-attachments/assets/03f9c748-cda9-418a-87ff-806dde05b0ba" />

Dataset:
- Source: NeelNanda/wiki-10k (Hugging Face)
- Size: 10,000 Wikipedia articles

Technologies:
- Python
- pandas
- Hugging Face datasets


Possible Improvements:
- Inverted index for faster query performance
- Cosine normalization using document vector norms
- Persistent storage of preprocessed data
