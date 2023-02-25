import sys
from common.util import preprocess
sys.path.append('..')

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)
