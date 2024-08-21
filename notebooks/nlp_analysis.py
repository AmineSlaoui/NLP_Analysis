"""
TODO: 
- Tokenize text, remove stopwords X
- Analyze and visualize abstracts data
    - Word Frequency Analysis: Determine the most common terms in the abstracts. X
    - Topic Modeling: Identify the main topics discussed in these articles.
    - Sentiment Analysis: Analyze the sentiment expressed in the abstracts.

For Topic Modeling need to:
    - Vectorize the Text X
    - Apply LDA
    - Display Topics
"""

"""
Notes:

Doc matrix prints in (i, j) k format where:
    - i refers to document number
    - j refers to term index (what word is being pointed)
    - k refers the number of time this word has appeared in the corresponding doc

"""

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


nltk.download('punkt_tab') # Identify boundaries between words (Tokenization)
nltk.download('stopwords') # Identify common stopwords 


# Tokenize each words then remove stopwords
def process_text(text): 
    processed_text = []
    if isinstance(text, str):
        text = text.lower()

        text = re.sub(r'\W+', ' ', text) # Gets rid of punctuation

        stop_words = set(stopwords.words('english'))

        tokens = word_tokenize(text)
        
        for words in tokens:
            if words not in stop_words:
                processed_text.append(words)

    return processed_text

# Find most common words in all abstracts
def common_word():
    flattened_lst = []
    for lst_words in data["processed_abstract"]:
        for word in lst_words:
            flattened_lst.append(word)

    counter = Counter(flattened_lst)
    return counter

# Process text
data = pd.read_csv("data/metadata.csv", nrows = 1000)
data["processed_abstract"] = data["abstract"].apply(process_text) 
data.to_csv("data/cleaned_data.csv", index = False)

mostcommon_words = common_word()

# Setup plotting through dataframe
words_df = pd.DataFrame(mostcommon_words.items(), columns=['Word', 'Frequency'])
words_df = words_df.sort_values(by='Frequency', ascending=False)
words_df = words_df.head(20)


# Vectorizing the text -> Transform text into a matrix of numerical values
data['processed_abstract_str'] = data['processed_abstract'].apply(lambda x: ' '.join(x))

# Ignore words that appear in more than 95% 
# Ignore words that appear in fewer than 2 documents
vectorizer = CountVectorizer(max_df=0.95, min_df=2)
doc_term_matrix = vectorizer.fit_transform(data['processed_abstract_str'])

# Applying Latent Dirichlet Allocation (LDA) Model to idenfity main topic in each article
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(doc_term_matrix)

# Display topics
words = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}:")
    print(" ".join([words[i] for i in topic.argsort()[-10:]]))
    print("\n")

# Plot most frequent words used in abstract
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Frequency', y='Word', data= words_df, palette='rocket', hue='Word')
# plt.title('Top 20 Most Frequent Words')
# plt.xlabel('Frequency')
# plt.ylabel('Words')
# plt.show()










