"""
TODO: 
- Tokenize text, remove stopwords X
- Analyze and visualize abstracts data
    - Word Frequency Analysis: Determine the most common terms in the abstracts. X
    - Topic Modeling: Identify the main topics discussed in these articles. X
    - Use TF-IDF to measure how important a term is within a document relative to all other docs
    - Sentiment Analysis: Analyze the sentiment expressed in the abstracts.

- For Topic Modeling need to:
    - Vectorize the Text X
    - Apply LDA X
    - Display Topics X

- For TF-IDF Modeling need to:
    - Implement TF-IDF Vectorization X 
    - Cluster documents into topics with K-Means X
    - Compare results with LDA model
    - Visualize 

    
"""

"""
Notes:

Count vectorizer transforms set of texts into matrix of token counts
Vectorizing the text -> Transform text into a matrix of numerical values

Doc matrix prints in (i, j) k format where:
    - i refers to document number
    - j refers to term index (what word is being pointed)
    - k refers the number of time this word has appeared in the corresponding doc

LDA model uses to find topics in set of documents
    - random_state ensures that the experiments is reproducible (running code several times)
    - lda.components is a matrix that represents importance of words in a topic 


LDA + TFIDF values: 
    - Ignore words that appear in more than 95% of documents
    - Ignore words that appear in fewer than 2 documents

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nltk.download('punkt_tab') # Identify boundaries between words (Tokenization)
nltk.download('stopwords') # Identify common stopwords 


# Tokenize each words then remove stopwords
def process_text(text): 
    processed_text = []
    if isinstance(text, str):
        text = text.lower()

        text = re.sub(r'\W+', ' ', text) # Gets rid of punctuation
        text = re.sub(r'\d+', '', text) # Gets rid of numbers

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


# Fetch top words in LDA model
def get_top_words_lda(model, feature_names, n_top_words=5):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        top_words.append(top_features)
    return top_words

# Fetch top words in TF-IDF + KMeans model
def get_top_words_kmeans(cluster_centers, feature_names, n_top_words=5):
    top_words = []
    for cluster_idx in range(cluster_centers.shape[0]):
        top_features_ind = cluster_centers[cluster_idx].argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        top_words.append(top_features)
    return top_words


# Process text
data = pd.read_csv("NLP_Analysis/data/metadata.csv", nrows = 1000)
data["processed_abstract"] = data["abstract"].apply(process_text) 
data.to_csv("NLP_Analysis/data/cleaned_data.csv", index = False)

mostcommon_words = common_word()

# Setup plotting through dataframe
words_df = pd.DataFrame(mostcommon_words.items(), columns=['Word', 'Frequency'])
words_df = words_df.sort_values(by='Frequency', ascending=False)
words_df = words_df.head(20)

data['processed_abstract_str'] = data['processed_abstract'].apply(lambda x: ' '.join(x))
data.to_csv("NLP_Analysis/data/cleaned_data.csv")


## LDA MODEL

# LDA Vectorization
lda_vectorizer = CountVectorizer(max_df=0.95, min_df=20)
lda_matrix = lda_vectorizer.fit_transform(data['processed_abstract_str'])

# Applying Latent Dirichlet Allocation (LDA) Model to idenfity main topic in each article
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(lda_matrix)
lda_words = lda_vectorizer.get_feature_names_out()

# Fetch words
lda_top_words = get_top_words_lda(lda_model, lda_words, 5)


## TF-IDF MODEL

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_df= 0.95, min_df=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_abstract_str'])

tfidf_words = tfidf_vectorizer.get_feature_names_out()

# Clustering
kmeans = KMeans(n_clusters= 10, random_state= 42)
kmeans.fit(tfidf_matrix)

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

# Fetch words
tfidf_top_words = get_top_words_kmeans(kmeans.cluster_centers_, tfidf_words, 5)

# Print top words for each topic
print("Top words for each LDA topic:")
for i, words in enumerate(lda_top_words):
    print(f"Topic {i + 1}: {', '.join(words)}")

print("\nTop words for each TF-IDF + K-Means topic:")
for i, words in enumerate(tfidf_top_words):
    print(f"Topic {i + 1}: {', '.join(words)}")


## Visualization

# Plot most frequent words used in abstract
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Frequency', y='Word', data= words_df, palette='rocket', hue='Word')
# plt.title('Top 20 Most Frequent Words')
# plt.xlabel('Frequency')
# plt.ylabel('Words')
# plt.show()

# Plot comparison chart between LDA and TFIDF models
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

for i in range(3):  
    # LDA topics
    axes[i, 0].barh(lda_top_words[i][::-1], range(1, 6))
    axes[i, 0].set_title(f"LDA Topic {i + 1}")
    
    # TF-IDF + K-Means topics
    axes[i, 1].barh(tfidf_top_words[i][::-1], range(1, 6))
    axes[i, 1].set_title(f"TF-IDF + K-Means Topic {i + 1}")

# plt.subplots_adjust()
plt.show()










