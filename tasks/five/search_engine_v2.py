import os
import re
import math
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

# Инициализация
lemmatizer = WordNetLemmatizer()

class SearchEngine:
    def __init__(self, pages_dir="pages", tfidf_dir="tf_idf_lemmas", max_pages=100):
        self.pages_dir = pages_dir
        self.max_pages = max_pages
        self.index = {}        # лемма -> {doc_id: tf_idf_weight}
        self.doc_vectors = {}  # doc_id -> {lemma: tf_idf_weight}
        self.doc_lengths = {}  # для нормализации векторов (длина вектора)
        
        self._load_tfidf_data(tfidf_dir)

    def _load_tfidf_data(self, tfidf_dir):
        print("Загрузка данных TF-IDF...")
        for i in range(1, self.max_pages + 1):
            filepath = os.path.join(tfidf_dir, f"{i}.txt")
            if not os.path.exists(filepath):
                continue
            
            self.doc_vectors[i] = {}
            sum_sq = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    lemma, idf, tfidf = parts[0], float(parts[1]), float(parts[2])
                    
                    self.doc_vectors[i][lemma] = tfidf
                    sum_sq += tfidf ** 2
                    
                    if lemma not in self.index:
                        self.index[lemma] = {}
                    self.index[lemma][i] = tfidf
            
            self.doc_lengths[i] = math.sqrt(sum_sq)

    def vector_search(self, query):
        # 1. Токенизация и лемматизация запроса
        query_words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        query_lemmas = [lemmatizer.lemmatize(w) for w in query_words]
        
        # 2. Формируем вектор запроса (простейший TF)
        query_vec = {}
        for l in query_lemmas:
            query_vec[l] = query_vec.get(l, 0) + 1
        
        q_sum_sq = sum(v**2 for v in query_vec.values())
        q_length = math.sqrt(q_sum_sq) if q_sum_sq > 0 else 1

        # 3. Считаем косинусное сходство со всеми документами
        scores = []
        for doc_id, doc_vec in self.doc_vectors.items():
            dot_product = 0
            for lemma, q_weight in query_vec.items():
                if lemma in doc_vec:
                    dot_product += q_weight * doc_vec[lemma]
            
            if dot_product > 0:
                # Формула косинусного сходства: (A * B) / (|A| * |B|)
                similarity = dot_product / (q_length * self.doc_lengths[doc_id])
                scores.append((doc_id, similarity))
        
        # Сортируем по убыванию сходства
        return sorted(scores, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    engine = SearchEngine()
    
    print("\n" + "="*30)
    print("ВЕКТОРНЫЙ ПОИСК ГОТОВ")
    print("="*30)
    
    while True:
        user_query = input("\nВведите поисковый запрос (или 'exit'): ")
        if user_query.lower() == 'exit': break
        
        results = engine.vector_search(user_query)
        
        if not results:
            print("Ничего не найдено.")
        else:
            print(f"Результаты (найдено {len(results)}):")
            for doc_id, score in results[:10]: # Топ-10 результатов
                print(f"Документ #{doc_id:3} | Релевантность: {score:.4f}")