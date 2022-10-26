from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd

model = SentenceTransformer('sberbank-ai/sbert_large_nlu_ru')

df = pd.read_csv('data.csv', index_col=0)


def main(q):
    startTime = time.time()
    embeddings = model.encode(df['all'].values.astype('U'))
    query = model.encode([q])
    cosine_similarity(embeddings, query)

    indx_val = np.argsort(cos_sim, axis=0)
    sorted_texts = []
    for i in range(len(indx_val) - 1, len(indx_val) - 12, -1):
        indx = indx_val[i]
        sorted_texts.append({'question': df.loc[[indx[0]], ['question']].values[0][0],
                             'comment': df.loc[[indx[0]], ['comment']].values[0][0],
                             'answer': df.loc[[indx[0]], ['answers']].values[0][0]})
    executionTime = (time.time() - startTime)
    return sorted_texts, executionTime