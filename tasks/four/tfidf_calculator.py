import os
import math
from collections import defaultdict
from pathlib import Path
import nltk

from tasks.two.nlp_processor import NLPProcessor, lemmatizer

class TFIDFCalculator:
    def __init__(
        self, 
        input_dir="pages_1", 
        output_dir_tokens="tf_idf_tokens", 
        output_dir_lemmas="tf_idf_lemmas"
    ):
        self.input_dir = input_dir
        self.output_dir_tokens = output_dir_tokens
        self.output_dir_lemmas = output_dir_lemmas
        # Инициализируем NLPProcessor для переиспользования его методов
        self.processor = NLPProcessor(input_dir=self.input_dir)
        
    def calculate(self):
        total_docs = len(os.listdir(self.input_dir))
        
        doc_tokens_list = {} # doc_id -> список валидных токенов (с дубликатами)
        doc_lemmas_list = {} # doc_id -> список валидных лемм (с дубликатами)
        
        term_df = defaultdict(int)  # термин -> количество документов
        lemma_df = defaultdict(int) # лемма -> количество документов
        
        print("Первый проход: сбор статистики (DF)...")
        for i in range(total_docs):
            doc_id = i + 1
            words = self.processor.extract_text(doc_id)
            if not words:
                continue
            
            valid_tokens = []
            valid_lemmas = []

            tagged_words = nltk.pos_tag(words)
            for token, pos_tag in tagged_words:
                if pos_tag in self.processor.STOP_TAGS:
                    continue
                
                valid_tokens.append(token)
                
                wn_pos = self.processor.get_wordnet_pos(pos_tag)
                lemma = lemmatizer.lemmatize(token, pos=wn_pos)
                valid_lemmas.append(lemma)
                
            doc_tokens_list[doc_id] = valid_tokens
            doc_lemmas_list[doc_id] = valid_lemmas
            
            # считаем DF (в скольких документах встретилось слово)
            # используем set() здесь, так как для DF нам важно только наличие слова в документе
            for unique_token in set(valid_tokens):
                term_df[unique_token] += 1
            for unique_lemma in set(valid_lemmas):
                lemma_df[unique_lemma] += 1
                
            print(f"Прочитана {doc_id}-ая страница.")
            
        print("\nВторой проход: расчет TF-IDF и сохранение...")
        Path(self.output_dir_tokens).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_lemmas).mkdir(parents=True, exist_ok=True)
        
        for doc_id in doc_tokens_list.keys():
            tokens = doc_tokens_list[doc_id]
            lemmas = doc_lemmas_list[doc_id]
            
            total_tokens = len(tokens)
            total_lemmas = len(lemmas)
            
            if total_tokens == 0 or total_lemmas == 0:
                continue
            
            # Подсчет частоты слов внутри текущего документа
            token_counts = defaultdict(int)
            for t in tokens: token_counts[t] += 1
            
            lemma_counts = defaultdict(int)
            for l in lemmas: lemma_counts[l] += 1
            
            with open(Path(self.output_dir_tokens) / f"{doc_id}.txt", "w", encoding="utf-8") as f:
                for token, count in token_counts.items():
                    tf = count / total_tokens
                    idf = math.log(total_docs / term_df[token])
                    tfidf = tf * idf
                    f.write(f"{token} {idf:.6f} {tfidf:.6f}\n")

            with open(Path(self.output_dir_lemmas) / f"{doc_id}.txt", "w", encoding="utf-8") as f:
                for lemma, count in lemma_counts.items():
                    tf = count / total_lemmas
                    idf = math.log(total_docs / lemma_df[lemma])
                    tfidf = tf * idf
                    f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")
                    
        print(f"Готово! Данные сохранены в '{self.output_dir_tokens}' и '{self.output_dir_lemmas}'.")

if __name__ == "__main__":
    calculator = TFIDFCalculator(input_dir="pages_1")
    calculator.calculate()