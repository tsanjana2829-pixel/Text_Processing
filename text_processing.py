import re
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

# 1. Download required tokenizers
nltk.download("punkt")
nltk.download("punkt_tab")

# 2. Read text file
with open(r"C:\Users\Windows\OneDrive\Desktop\text processing\dataset\rawtext.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Original Text:\n")
print(text)

# 3. SENTENCE TOKENIZATION (Must be done before removing punctuation)
sentences = sent_tokenize(text)
print("\nSentence Tokens:")
for sentence in sentences:
    print(sentence)

# 4. CLEAN TEXT (For word tokenization)
clean_text = text.lower()  # Convert to lowercase
clean_text = re.sub(r"\d+", "", clean_text)  # Remove numbers
clean_text = re.sub(
    r"[^\w\s]", "", clean_text
)  # Remove punctuation and special characters
clean_text = re.sub(r"\s+", " ", clean_text).strip()  # Remove extra spaces

print("\nCleaned Text:\n")
print(clean_text)

# 5. WORD TOKENIZATION
words = word_tokenize(clean_text)
print("\nWord Tokens:")
print(words)

# 6. Save output files
pd.DataFrame({"Clean_Text": [clean_text]}).to_csv(
    "dataset/clean_text.csv", index=False
)
pd.DataFrame({"Token": words}).to_csv("dataset/tokens.csv", index=False)

print("\nFiles saved successfully.")