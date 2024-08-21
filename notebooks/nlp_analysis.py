"""
TODO: 
- Tokenize text, remove stopwords X
- Analyze abstracts
    - Word Frequency Analysis: Determine the most common terms in the abstracts.
    - Topic Modeling: Identify the main topics discussed in these articles.
    - Sentiment Analysis: Analyze the sentiment expressed in the abstracts.
- Create wordcloud or bar chart to visualize 
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt_tab') # Identify boundaries between words (Tokenization)
nltk.download('stopwords') # Identify common stopwords 

data = pd.read_csv("NLP_Analysis/data/metadata.csv", nrows = 1000)


# Tokenize each words then remove stopwords
def process_text(text): # Text is in string format
    processed_text = []
    if isinstance(text, str):
        text = text.lower()

        text = re.sub(r'\W+', ' ', text) # Gets rid of punctuation

        stop_words = set(stopwords.words('english'))

        tokens = word_tokenize(text)
        
        for words in tokens:
            if words not in stop_words:
                processed_text.append(words)

    # print(processed_text)
    return processed_text


def common_word(): # Text is in list format
    flattened_lst = []
    for lst_words in data["processed_abstract"]:
        for word in lst_words:
            flattened_lst.append(word)

    counter = Counter(flattened_lst)
    return counter

# Process text
data["processed_abstract"] = data["abstract"].apply(process_text) 
data.to_csv("NLP_Analysis/data/cleaned_data_with_processed_abstract.csv", index = False)

mostcommon_words = common_word()

# Easier to plot when it is converted to dataframe
words_df = pd.DataFrame(mostcommon_words.items(), columns=['Word', 'Frequency'])
words_df = words_df.sort_values(by='Frequency', ascending=False)
words_df = words_df.head(20)

print(words_df.shape)


plt.figure(figsize=(12, 6))
sns.barplot(x='Frequency', y='Word', data= words_df, palette='rocket', hue='Word')
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()





