import os
import re
import math
import nltk

from tasks.two.nlp_processor import NLPProcessor, lemmatizer

class VectorSearchEngine:
    def __init__(
        self, 
        tfidf_dir="tf_idf_lemmas", 
        index_file="index.txt"
    ):
        self.tfidf_dir = tfidf_dir
        self.index_file = index_file
        
        self.doc_vectors = {}  # doc_id -> {lemma: tf_idf_weight}
        self.doc_lengths = {}  # doc_id -> длина вектора (для косинусного сходства)
        self.url_map = {}      # doc_id -> url
        
        self._load_data()

    def _load_data(self):
        print("Загрузка данных TF-IDF и ссылок...")
        
        # 1. Загружаем карту ссылок
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        self.url_map[int(parts[0])] = parts[1]

        # 2. Загружаем векторы из папки tf_idf_lemmas
        if not os.path.exists(self.tfidf_dir):
            print(f"Ошибка: Папка {self.tfidf_dir} не найдена. Сначала запустите TF-IDF (Задание 4).")
            return

        for filename in os.listdir(self.tfidf_dir):
            if not filename.endswith(".txt"):
                continue
                
            doc_id = int(filename.split(".")[0])
            filepath = os.path.join(self.tfidf_dir, filename)
            
            self.doc_vectors[doc_id] = {}
            sum_sq = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        lemma = parts[0]
                        tfidf = float(parts[2])
                        
                        self.doc_vectors[doc_id][lemma] = tfidf
                        sum_sq += tfidf ** 2
            
            # Считаем длину вектора документа
            self.doc_lengths[doc_id] = math.sqrt(sum_sq) if sum_sq > 0 else 1
            
        print(f"Успешно загружены векторы для {len(self.doc_vectors)} документов.")

    def search(self, query):
        """Выполняет векторный поиск по запросу."""
        # 1. Извлекаем слова из запроса
        words = re.findall(r'\b[a-zA-Z]+\b', query)
        words = [w.lower() for w in words]
        
        # 2. Обрабатываем запрос через методы NLPProcessor (ПЕРЕИСПОЛЬЗОВАНИЕ)
        query_lemmas = []
        tagged_words = nltk.pos_tag(words)
        for token, pos_tag in tagged_words:
            # Отсеиваем предлоги и союзы из запроса так же, как в текстах
            if pos_tag in NLPProcessor.STOP_TAGS:
                continue
            
            wn_pos = NLPProcessor.get_wordnet_pos(pos_tag)
            lemma = lemmatizer.lemmatize(token, pos=wn_pos)
            query_lemmas.append(lemma)
            
        if not query_lemmas:
            return []

        # 3. Формируем вектор запроса (TF)
        query_vec = {}
        for l in query_lemmas:
            query_vec[l] = query_vec.get(l, 0) + 1
            
        q_len = math.sqrt(sum(v**2 for v in query_vec.values())) if query_vec else 1

        # 4. Считаем косинусное сходство
        results = []
        for doc_id, doc_vec in self.doc_vectors.items():
            dot_product = 0
            for lemma, q_weight in query_vec.items():
                if lemma in doc_vec:
                    dot_product += q_weight * doc_vec[lemma]
            
            if dot_product > 0:
                similarity = dot_product / (q_len * self.doc_lengths[doc_id])
                results.append({
                    "doc_id": doc_id,
                    "url": self.url_map.get(doc_id, f"Document #{doc_id}"),
                    "score": similarity
                })
                
        # Сортируем по убыванию релевантности
        return sorted(results, key=lambda x: x["score"], reverse=True)

def start_interactive_search(engine: VectorSearchEngine):
    print("\n" + "="*40)
    print("ВЕКТОРНЫЙ ПОИСК ГОТОВ")
    print("="*40)
    
    while True:
        query = input("\nВведите запрос (или 'exit'): ")
        if query.lower() == 'exit':
            break
            
        results = engine.search(query)
        
        if not results:
            print("❌ Ничего не найдено. Попробуйте другие ключевые слова.")
        else:
            print(f"✅ Найдено результатов: {len(results)}")
            # Показываем Топ-10
            for rank, res in enumerate(results[:10], 1):
                print(f"{rank}. [{res['score']:.4f}] Doc #{res['doc_id']:<3} | {res['url']}")