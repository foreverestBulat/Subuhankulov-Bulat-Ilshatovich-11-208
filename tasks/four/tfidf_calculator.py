import os
import re
import math
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def calculate_tfidf(input_dir="pages", max_pages=100):
    print("Инициализация NLTK...")
    lemmatizer = WordNetLemmatizer()
    stop_tags = {'IN', 'CC', 'DT', 'TO', 'PRP', 'PRP$'}

    # Структуры для хранения данных
    doc_tokens = {} # doc_id -> список токенов в документе
    doc_lemmas = {} # doc_id -> список лемм в документе
    
    term_df = {}    # термин -> в скольких документах встречается
    lemma_df = {}   # лемма -> в скольких документах встречается
    
    # Создаем папки для результатов
    out_tokens_dir = "tf_idf_tokens"
    out_lemmas_dir = "tf_idf_lemmas"
    os.makedirs(out_tokens_dir, exist_ok=True)
    os.makedirs(out_lemmas_dir, exist_ok=True)
    
    print("Чтение документов и сбор статистики (DF)...")
    
    # ПЕРВЫЙ ПРОХОД: Считываем все тексты и считаем DF (Document Frequency)
    for i in range(1, max_pages + 1):
        filepath = os.path.join(input_dir, f"{i}.txt")
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            text = BeautifulSoup(f.read(), 'html.parser').get_text(separator=' ')
            
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        tagged_words = nltk.pos_tag(words)
        
        valid_tokens = []
        valid_lemmas = []
        
        for token, pos_tag in tagged_words:
            if pos_tag in stop_tags:
                continue
            
            valid_tokens.append(token)
            
            wn_pos = get_wordnet_pos(pos_tag)
            lemma = lemmatizer.lemmatize(token, pos=wn_pos)
            valid_lemmas.append(lemma)
            
        doc_tokens[i] = valid_tokens
        doc_lemmas[i] = valid_lemmas
        
        # Обновляем DF (считаем уникальные слова в текущем документе)
        for unique_token in set(valid_tokens):
            term_df[unique_token] = term_df.get(unique_token, 0) + 1
            
        for unique_lemma in set(valid_lemmas):
            lemma_df[unique_lemma] = lemma_df.get(unique_lemma, 0) + 1
            
    total_docs = len(doc_tokens)
    print(f"Обработано {total_docs} документов. Начинаю расчет TF-IDF...")
    
    # ВТОРОЙ ПРОХОД: Считаем TF и TF-IDF, затем записываем в файлы
    for doc_id in doc_tokens.keys():
        tokens = doc_tokens[doc_id]
        lemmas = doc_lemmas[doc_id]
        
        total_tokens = len(tokens)
        total_lemmas = len(lemmas)
        
        if total_tokens == 0:
            continue
            
        # Подсчет вхождений в текущем документе
        token_counts = {}
        for t in tokens: token_counts[t] = token_counts.get(t, 0) + 1
        
        lemma_counts = {}
        for l in lemmas: lemma_counts[l] = lemma_counts.get(l, 0) + 1
        
        # Запись файлов для ТОКЕНОВ
        with open(os.path.join(out_tokens_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            for token, count in token_counts.items():
                tf = count / total_tokens
                idf = math.log(total_docs / term_df[token])
                tfidf = tf * idf
                # Формат: <термин> <idf> <tf-idf>
                f.write(f"{token} {idf:.6f} {tfidf:.6f}\n")
                
        # Запись файлов для ЛЕММ
        with open(os.path.join(out_lemmas_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            for lemma, count in lemma_counts.items():
                tf = count / total_lemmas
                idf = math.log(total_docs / lemma_df[lemma])
                tfidf = tf * idf
                # Формат: <лемма> <idf> <tf-idf>
                f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")
                
    print("Готово! Папки 'tf_idf_tokens' и 'tf_idf_lemmas' успешно созданы и заполнены.")

if __name__ == "__main__":
    calculate_tfidf(input_dir="pages", max_pages=100)