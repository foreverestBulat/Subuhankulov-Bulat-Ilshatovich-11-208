import os
import re
from bs4 import BeautifulSoup
import pymorphy3
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Скачиваем необходимые словари и модели NLTK (при первом запуске)
nltk.download('averaged_perceptron_tagger_eng', quiet=True) # для новых версий nltk
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class NLPProcessor:
    def __init__(
        self,
        input_dir="pages",
        tokens_path="tokens.txt",
        lemmas_path="lemmas.txt",
        max_pages=100
    ):
        self.input_dir = input_dir
        self.tokens_path = tokens_path
        self.lemmas_path = lemmas_path
        self.max_pages = max_pages


    def process_texts_ru(self):
        print("Инициализация морфологического анализатора...")
        morph = pymorphy3.MorphAnalyzer()
        
        all_tokens = set()
        lemma_dict = {}  # Словарь формата: лемма -> множество(токены)
        
        # Части речи, которые мы исключаем (Предлоги, Союзы, Частицы, Междометия)
        stop_tags = {'PREP', 'CONJ', 'PRCL', 'INTJ'}
        
        print("Начинаю обработку файлов...")
        
        for i in range(1, self.max_pages + 1):
            filepath = os.path.join(self.input_dir, f"{i}.txt")
            if not os.path.exists(filepath):
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            # 1. Очистка от HTML-разметки
            # BeautifulSoup извлекает только видимый текст
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ')
            
            # 2. Очистка от мусора и чисел (Токенизация)
            # Регулярное выражение \b[а-яёА-ЯЁ]+\b ищет только слова, 
            # состоящие ИСКЛЮЧИТЕЛЬНО из русских букв.
            # Это автоматически исключает числа, английские слова, email и слова с цифрами (вроде "covid19").
            words = re.findall(r'\b[а-яёА-ЯЁ]+\b', text)
            
            for word in words:
                token = word.lower()
                
                # 3. Морфологический анализ токена
                parsed = morph.parse(token)[0]
                
                # Если это союз, предлог, частица и т.д. — пропускаем
                if parsed.tag.POS in stop_tags:
                    continue
                    
                # Добавляем уникальный токен в общее множество
                all_tokens.add(token)
                
                # 4. Группировка по леммам
                lemma = parsed.normal_form
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = set()
                lemma_dict[lemma].add(token)
                
            if i % 10 == 0:
                print(f"Обработано {i}/{self.max_pages} файлов...")
                
        print("Сохранение результатов в файлы...")
        
        # Сохраняем tokens.txt (каждый токен с новой строки)
        with open(self.tokens_path, "w", encoding="utf-8") as f:
            for t in sorted(all_tokens):
                f.write(f"{t}\n")
                
        # Сохраняем lemmas.txt (лемма и список ее токенов)
        with open(self.lemmas_path, "w", encoding="utf-8") as f:
            for lemma, tokens_set in sorted(lemma_dict.items()):
                tokens_str = " ".join(sorted(tokens_set))
                f.write(f"{lemma} {tokens_str}\n")
                
        print(f"Готово! Файлы '{self.tokens_path}' и '{self.lemmas_path}' успешно созданы.")


    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """Конвертирует теги частей речи NLTK в формат, понятный лемматизатору"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN # По умолчанию считаем существительным

    def process_english_texts(self):
        print("Инициализация NLTK лемматизатора...")
        lemmatizer = WordNetLemmatizer()
        
        all_tokens = set()
        lemma_dict = {}
        
        # Теги частей речи в NLTK, которые мы считаем "мусором" по заданию:
        # IN - предлоги и подчинительные союзы
        # CC - сочинительные союзы
        # DT - артикли/определители (the, a, an)
        # TO - частица to
        # PRP, PRP$ - местоимения
        stop_tags = {'IN', 'CC', 'DT', 'TO', 'PRP', 'PRP$'}
        
        print("Начинаю обработку файлов...")
        
        for i in range(1, self.max_pages + 1):
            filepath = os.path.join(self.input_dir, f"{i}.txt")
            if not os.path.exists(filepath):
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            # 1. Очистка от HTML-разметки
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ')
            
            # 2. Очистка от мусора
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            words_lower = [w.lower() for w in words]
            
            # части речи
            tagged_words = nltk.pos_tag(words_lower)
            
            for token, pos_tag in tagged_words:
                # Отсеиваем предлоги, союзы, артикли
                if pos_tag in stop_tags:
                    continue
                    
                # Добавляем в список токенов
                all_tokens.add(token)
                
                # Лемматизация
                wn_pos = self.get_wordnet_pos(pos_tag)
                lemma = lemmatizer.lemmatize(token, pos=wn_pos)
                
                # Группировка
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = set()
                lemma_dict[lemma].add(token)
                
            if i % 10 == 0:
                print(f"Обработано {i}/{self.max_pages} файлов...")
                
        print("Сохранение результатов в файлы...")
        
        with open(self.tokens_path, "w", encoding="utf-8") as f:
            for t in sorted(all_tokens):
                f.write(f"{t}\n")
                
        with open(self.lemmas_path, "w", encoding="utf-8") as f:
            for lemma, tokens_set in sorted(lemma_dict.items()):
                tokens_str = " ".join(sorted(tokens_set))
                f.write(f"{lemma} {tokens_str}\n")
                
        print("Готово! Файлы 'tokens.txt' и 'lemmas.txt' обновлены для английского языка.")



if __name__ == "__main__":
    # Запускаем обработку для 100 скачанных страниц
    # # ru
    # processor = NLPProcessor(input_dir="pages", tokens_path="tokens.txt", lemmas_path="lemmas.txt", max_pages=100)
    # processor.process_texts_ru()
    
    # en
    processor = NLPProcessor(input_dir="pages", tokens_path="tokens_en.txt", lemmas_path="lemmas_en.txt", max_pages=100)
    processor.process_english_texts()