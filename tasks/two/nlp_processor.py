import json
import os
from pathlib import Path
import re
from collections import defaultdict

from bs4 import BeautifulSoup
from nltk.corpus import wordnet
import nltk



# Скачиваем необходимые словари и модели NLTK (при первом запуске)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = nltk.WordNetLemmatizer()


class NLPProcessor:
    # Теги частей речи в NLTK, которые мы считаем "мусором" по заданию:
    # IN - предлоги и подчинительные союзы
    # CC - сочинительные союзы
    # DT - артикли/определители (the, a, an)
    # TO - частица to
    # PRP, PRP$ - местоимения
    STOP_TAGS = {'IN', 'CC', 'DT', 'TO', 'PRP', 'PRP$'}
    
    def __init__(
        self,
        input_dir="pages",
        output_dir="data"
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def extract_text(self, i):
        """Очищает текст i-го файла от разметри и мусора."""
        filepath = os.path.join(self.input_dir, f"{i}.txt")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ')
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return [w.lower() for w in words]

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """Конвертирует теги частей речи NLTK в формат, понятный лемматизатору."""
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

    def process_tokens_and_lemmas(self, words):
        """Создание токенов и лемм."""
        tokens = set()
        lemmas = defaultdict(set)
        tagged_words = nltk.pos_tag(words)
        for token, pos_tag in tagged_words:
            if pos_tag in self.STOP_TAGS:
                continue
            tokens.add(token)
            wn_pos = self.get_wordnet_pos(pos_tag)
            lemma = lemmatizer.lemmatize(token, pos=wn_pos)
            lemmas[lemma].add(token)
        return tokens, lemmas

    def process(self):
        for i in range(len(os.listdir(self.input_dir))):
            words = self.extract_text(i + 1)
            tokens, lemmas = self.process_tokens_and_lemmas(words)

            path = Path(self.output_dir) / "tokens"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / f"{i + 1}.txt", "w", encoding="utf-8") as file:
                file.write('\n'.join(tokens))
            path = Path(self.output_dir) / "lemmas"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / f"{i + 1}.json", "w", encoding="utf-8") as file:
                # json не может set сохранить, поэтому в list
                json.dump({k: list(v) for k, v in lemmas.items()}, file, indent=4)
            print(f"Обработана {i + 1}-я страница")


if __name__ == "__main__":
    processor = NLPProcessor(input_dir="pages_1", output_dir="data_1")
    processor.process()