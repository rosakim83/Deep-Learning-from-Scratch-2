import sys
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity
sys.path.append('..')


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어의 단어 벡터를 꺼낸다
    if query not in word_to_id:
        print(f'{query}을(를) 찾을 수 없습니다.')
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 검색어의 단어 벡터와 다른 모든 단어 벡터와의 코사인 유사도를 각각 계산한다
    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력한다
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f' {id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)
