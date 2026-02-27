import os
import re
import json
from collections import defaultdict

from tasks.two.nlp_processor import NLPProcessor, lemmatizer

class SearchEngine:
    def __init__(
        self,
        input_dir="pages",
        output_file="inverted_index.json"
    ):
        self.input_dir = input_dir
        self.output_file = output_file
        
    def build_inverted_index(self):
        inverted_index = defaultdict(set)
        processor = NLPProcessor(input_dir=self.input_dir)
        
        total_docs = len(os.listdir(processor.input_dir))
        for i in range(total_docs):
            doc_id = i + 1
            words = processor.extract_text(doc_id)
            if not words:
                continue
            tokens, lemmas = processor.process_tokens_and_lemmas(words)

            for lemma in lemmas.keys():
                inverted_index[lemma].add(doc_id)
                
            print(f"Обработана {doc_id}-ая страница.")
    
        with open(self.output_file, "w", encoding="utf-8") as file:
            json_dict = {
                lemma: sorted(list(doc_ids)) 
                for lemma, doc_ids in sorted(inverted_index.items())
            }
            json.dump(json_dict, file, indent=4, ensure_ascii=False)
            
        print(f"Создан файл {self.output_file}")
        return inverted_index


def parse_query_to_postfix(query):
    """Преобразует строку запроса в обратную польскую запись"""
    tokens = re.findall(r'\(|\)|AND|OR|NOT|[a-zA-Z]+', query)
    output = []
    ops = []
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}
    
    for token in tokens:
        if token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            ops.pop()
        elif token in precedence:
            while ops and precedence[ops[-1]] >= precedence[token]:
                output.append(ops.pop())
            ops.append(token)
        else:
            output.append(token)
            
    while ops:
        output.append(ops.pop())
        
    return output


def evaluate_postfix(postfix_query, inverted_index, total_docs):
    """Вычисляет результат запроса используя множества (sets)"""
    stack = []
    all_docs = set(range(1, total_docs + 1))
    
    try:
        for token in postfix_query:
            if token == 'AND':
                set2, set1 = stack.pop(), stack.pop()
                stack.append(set1 & set2)
            elif token == 'OR':
                set2, set1 = stack.pop(), stack.pop()
                stack.append(set1 | set2)
            elif token == 'NOT':
                set1 = stack.pop()
                stack.append(all_docs - set1)
            else:
                lemma = lemmatizer.lemmatize(token.lower())
                docs = set(inverted_index.get(lemma, set()))
                stack.append(docs)
                
        return stack[0] if stack else set()
    except IndexError:
        print("Ошибка: Некорректный синтаксис запроса.")
        return set()


def start(engine: SearchEngine, inverted_index: defaultdict):
    print("\nВведите булев запрос на английском (например: (cat AND dog) OR NOT bird).")
    print("Для выхода введите 'exit'.")
    
    total_docs = len(os.listdir(engine.input_dir))
    
    while True:
        query = input("\nВаш запрос: ")
        if query.lower() == 'exit':
            break
            
        postfix = parse_query_to_postfix(query)
        result = evaluate_postfix(postfix, inverted_index, total_docs)
        
        if result:
            print(f"✅ Найдено в документах: {sorted(list(result))}")
        else:
            print("❌ По вашему запросу ничего не найдено.")


if __name__ == "__main__":
    engine = SearchEngine(input_dir="pages_1", output_file="inverted_index_1.json")
    inverted_index = engine.build_inverted_index()
    start(engine, inverted_index)