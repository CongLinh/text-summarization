# -*- coding: utf-8 -*-
print("LinhNVC")
import nltk
from gensim.models import KeyedVectors 
from pyvi import ViTokenizer
import numpy as np
import input_paragraph

w2v = KeyedVectors.load_word2vec_format("vi_txt/vi.vec")
vocab = w2v.vocab #Danh sách các từ trong từ điển


# TEST 
my_sentence = "là và của"
my_sentence_tokenized = ViTokenizer.tokenize(my_sentence)
print("========================\n", my_sentence_tokenized)
words = my_sentence_tokenized.split(" ")
print("========================\n", words)
my_sentence_vec = np.zeros((100))
print("========================\n", my_sentence_vec)
for word in words:
    if word in vocab:
        my_sentence_vec+=w2v[word]
    # X.append(sentence_vec)

print(my_sentence_vec)


paragraph = input_paragraph.content
print(paragraph)









