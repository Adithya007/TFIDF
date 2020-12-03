from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import gensim
model = gensim.models.Word2Vec.load('./bbcmodellower')
import os

folder1 = os.listdir('./bbc 2/business/')
folder2 = os.listdir('./bbc 2/entertainment')
folder3 = os.listdir('./bbc 2/politics')
folder4 = os.listdir('./bbc 2/sport')
folder5 = os.listdir('./bbc 2/tech')

def remove_stopwords(example_sent):
    stop_words = set(stopwords.words('english'))  
    word_tokens = example_sent.split(' ')
    filtered_sentence = [w for w in word_tokens if not w in stop_words]    
    filtered_sentence = []  
    for w in word_tokens:  
        if w not in stop_words:  
            filtered_sentence.append(w)  
    fitered_string = ' '.join(filtered_sentence)
    return fitered_string.strip()

def text_preprocess(file_name):
    string = file_name.read()
    list_of_words = string.split('.')
    list_of_words_without_stop_words = []
    for sentence in list_of_words:
        new_word = remove_stopwords(sentence)
        list_of_words_without_stop_words.append(new_word)
    final_list = []
    for word in list_of_words_without_stop_words:
        string_replaced = re.sub("[^a-zA-Z]+", " ", word)
        final_list.append(string_replaced.strip().lower())
    return final_list

full_list_of_folder = []
c = 0
try:
    for file_name in folder1:
        print (str(file_name))
        fi = open("./bbc 2/business/" + file_name, "r")
        list_1 = text_preprocess(fi)
        for element in list_1:
                full_list_of_folder.append(element)
    for file_name in folder2:
        print (str(file_name))
        fi = open("./bbc 2/entertainment/" + file_name, "r")
        list_1 = text_preprocess(fi)
        for element in list_1:
                full_list_of_folder.append(element)
    for file_name in folder3:
        print (str(file_name))
        fi = open("./bbc 2/politics/" + file_name, "r")
        list_1 = text_preprocess(fi)
        for element in list_1:
                full_list_of_folder.append(element)
    for file_name in folder4:
        print (str(file_name))
        fi = open("./bbc 2/sport/" + file_name, "r")
        list_1 = text_preprocess(fi)
        for element in list_1:
                full_list_of_folder.append(element)
    for file_name in folder5:
        print (str(file_name))
        fi = open("./bbc 2/tech/" + file_name, "r")
        list_1 = text_preprocess(fi)
        for element in list_1:
                full_list_of_folder.append(element)
except:
    c= c + 1

for element in full_list_of_folder:
    if len(element) == 0:
        full_list_of_folder.remove(element)

full_list_of_folder

documents_df = pd.DataFrame()
documents_df['documents_cleaned'] = full_list_of_folder

####Tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(documents_df.documents_cleaned)
tokenized_documents=tokenizer.texts_to_sequences(documents_df.documents_cleaned)
tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=64,padding='post')
vocab_size=len(tokenizer.word_index)+1


# creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
embedding_matrix=np.zeros((vocab_size,300))
for word,i in tokenizer.word_index.items():
    if word in model:
        embedding_matrix[i]=model[word]
# creating document-word embeddings
document_word_embeddings=np.zeros((len(tokenized_paded_documents),64,300))
for i in range(len(tokenized_paded_documents)):
    for j in range(len(tokenized_paded_documents[0])):
        document_word_embeddings[i][j]=embedding_matrix[tokenized_paded_documents[i][j]]


tfidfvectoriser=TfidfVectorizer()
tfidfvectoriser.fit_transform(documents_df.documents_cleaned)
tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)

# calculating average of word vectors of a document weighted by tf-idf
document_embeddings=np.zeros((len(tokenized_paded_documents),300))
words=tfidfvectoriser.get_feature_names()
for i in range(len(document_word_embeddings)):
    for j in range(len(words)):
        document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i][j]
print (document_embeddings.shape)
pairwise_similarities=cosine_similarity(document_embeddings)
pairwise_differences=euclidean_distances(document_embeddings)
