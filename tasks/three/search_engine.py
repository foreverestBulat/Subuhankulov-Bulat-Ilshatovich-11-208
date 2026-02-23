import os
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

def build_inverted_index(input_dir="pages", max_pages=100):
    print("Построение инвертированного индекса...")
    inverted_index = {}
    
    for doc_id in range(1, max_pages + 1):
        filepath = os.path.join(input_dir, f"{doc_id}.txt")
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            text = BeautifulSoup(f.read(), 'html.parser').get_text(separator=' ')
            
        # Извлекаем английские слова и переводим в нижний регистр
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        for word in words:
            # Лемматизируем слово перед добавлением в индекс
            lemma = lemmatizer.lemmatize(word)
            
            if lemma not in inverted_index:
                inverted_index[lemma] = set()
            inverted_index[lemma].add(doc_id)
            
    # Сохраняем индекс в файл
    print("Сохранение индекса в файл inverted_index.txt...")
    with open("inverted_index.txt", "w", encoding="utf-8") as f:
        for lemma, doc_ids in sorted(inverted_index.items()):
            # Преобразуем множество ID в строку: 1, 5, 42
            ids_str = ", ".join(map(str, sorted(doc_ids)))
            f.write(f"{lemma}: {ids_str}\n")
            
    return inverted_index

def parse_query_to_postfix(query):
    """Преобразует строку запроса в обратную польскую запись (алгоритм сортировочной станции)"""
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
            ops.pop() # удаляем '('
        elif token in precedence: # Если это оператор
            while ops and precedence[ops[-1]] >= precedence[token]:
                output.append(ops.pop())
            ops.append(token)
        else: # Если это обычное слово
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
                stack.append(set1 & set2) # Пересечение
            elif token == 'OR':
                set2, set1 = stack.pop(), stack.pop()
                stack.append(set1 | set2) # Объединение
            elif token == 'NOT':
                set1 = stack.pop()
                stack.append(all_docs - set1) # Разность
            else:
                # Лемматизируем слово из запроса, чтобы оно совпало с ключом в индексе
                lemma = lemmatizer.lemmatize(token.lower())
                docs = inverted_index.get(lemma, set())
                stack.append(docs)
                
        return stack[0] if stack else set()
    except IndexError:
        print("Ошибка: Некорректный синтаксис запроса.")
        return set()

if __name__ == "__main__":
    TOTAL_PAGES = 100
    
    # 1. Строим индекс
    index = build_inverted_index(input_dir="pages", max_pages=TOTAL_PAGES)
    print("Готово! Индекс построен.\n" + "="*40)
    
    # 2. Интерактивный поиск
    print("Введите булев запрос на английском (например: (cat AND dog) OR NOT bird).")
    print("Для выхода введите 'exit'.")
    
    while True:
        query = input("\nВаш запрос: ")
        if query.lower() == 'exit':
            break
            
        postfix = parse_query_to_postfix(query)
        result = evaluate_postfix(postfix, index, TOTAL_PAGES)
        
        if result:
            print(f"✅ Найдено в документах: {sorted(list(result))}")
        else:
            print("❌ По вашему запросу ничего не найдено.")