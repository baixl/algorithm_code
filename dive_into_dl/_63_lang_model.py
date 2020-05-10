#coding:utf-8
import  torch
with open('jaychou_lyrics.txt', 'r') as f:
      corpus_chars = f.read()
corpus_chars = corpus_chars.replace("\n", " ").replace("\r", " ")
print(corpus_chars[:1000])

idx_2_char = list(set(corpus_chars))
print(len(idx_2_char))
char_2_idx = dict([(char, i)  for i, char in enumerate(idx_2_char)])
vocab_size = len(char_2_idx)
print(vocab_size)


corpus_indices = [char_2_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_2_char[idx] for idx in sample]))
print('indices:', sample)
