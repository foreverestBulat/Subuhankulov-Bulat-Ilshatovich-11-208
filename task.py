import argparse
import json

from tasks.one.crawler import Crawler
from tasks.two.nlp_processor import NLPProcessor
from tasks.three.search_engine import SearchEngine, start
from tasks.four.tfidf_calculator import TFIDFCalculator
from tasks.five.search_engine_v2 import VectorSearchEngine, start_interactive_search


class TaskScripts:
    @staticmethod
    def run_crawler(args):
        with open("tasks/one/links.json", "r", encoding="utf-8") as file:
            urls = json.load(file)
            
        crawler = Crawler(
            output_dir=args.output_dir,
            index_file=args.index_file
        )
        crawler.run_crawler_from_list(urls)
    
    @staticmethod
    def run_nlp(args):
        processor = NLPProcessor(input_dir=args.input_dir, output_dir=args.output_dir)
        processor.process()

    @staticmethod
    def run_search_engine(args):
        engine = SearchEngine(input_dir=args.input_dir, output_file=args.output_file)
        inverted_index = engine.build_inverted_index()
        start(engine, inverted_index)
    
    @staticmethod
    def run_tfidf(args):
        calculator = TFIDFCalculator(
            input_dir=args.input_dir,
            output_dir_tokens=args.output_tokens,
            output_dir_lemmas=args.output_lemmas
        )
        calculator.calculate()

    @staticmethod
    def run_vector_search(args):
        engine = VectorSearchEngine(
            tfidf_dir=args.tfidf_dir,
            index_file=args.index_file
        )
        start_interactive_search(engine)

def main():
    parser = argparse.ArgumentParser(
        description="Менеджер задач поисковой системы. Управляет всеми этапами пайплайна."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")
    subparsers.required = True

    # === Задание 1: Краулер ===
    crawl_parser = subparsers.add_parser("crawl", help="Запустить скачивание страниц (Задание 1)")
    crawl_parser.add_argument(
        "-od", "--output-dir", 
        required=True,
        help="Путь до папки сохранения страниц."
    )
    crawl_parser.add_argument(
        "-if", "--index-file", 
        required=True, 
        help="Файл сохранения 'Выкачки'."
    )
    crawl_parser.set_defaults(func=TaskScripts.run_crawler)
    
    # === Задание 2: NLP и Лемматизация ===
    nlp_parser = subparsers.add_parser("nlp", help="Извлечь токены и леммы (Задание 2)")
    nlp_parser.add_argument(
        "-id", "--input-dir",
        required=True,
        help="Путь до папки со страницами."
    )
    nlp_parser.add_argument(
        "-od", "--output-dir", 
        required=True,
        help="Путь до папки сохранения токенов и лемм."
    )
    nlp_parser.set_defaults(func=TaskScripts.run_nlp)
    
    # === Задание 3: Инвертированный индекс ===
    index_parser = subparsers.add_parser("index", help="Построить инвертированный индекс (Задание 3)")
    index_parser.add_argument(
        "-id", "--input-dir", 
        required=True,
        help="Путь до папки со страницами."
    )
    index_parser.add_argument(
        "-of", "--output-file", 
        required=True,
        help="Название файла для сохранения."
    )
    index_parser.set_defaults(func=TaskScripts.run_search_engine)

    # === Задание 4: TF-IDF ===
    tfidf_parser = subparsers.add_parser("tfidf", help="Рассчитать TF, IDF и TF-IDF (Задание 4)")
    tfidf_parser.add_argument("-id", "--input-dir", required=True, help="Путь до папки со страницами.")
    tfidf_parser.add_argument("-ot", "--output-tokens", default="tf_idf_tokens", help="Папка для токенов.")
    tfidf_parser.add_argument("-ol", "--output-lemmas", default="tf_idf_lemmas", help="Папка для лемм.")
    tfidf_parser.set_defaults(func=TaskScripts.run_tfidf)

    # === Задание 5: Векторный поиск ===
    search_parser = subparsers.add_parser("search", help="Запустить векторный поиск (Задание 5)")
    search_parser.add_argument(
        "-td", "--tfidf-dir", 
        default="tf_idf_lemmas", 
        help="Путь к папке с подсчитанными TF-IDF лемм."
    )
    search_parser.add_argument(
        "-if", "--index-file", 
        default="index.txt", 
        help="Файл со ссылками выкачки."
    )
    search_parser.set_defaults(func=TaskScripts.run_vector_search)
    
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()