import os
import re
import math
import requests
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")
lemmatizer = WordNetLemmatizer()

# --- ЛОГИКА ПОИСКОВОГО ДВИЖКА ---

def get_data():
    url_map = {}
    if os.path.exists("index.txt"):
        with open("index.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    url_map[int(parts[0])] = parts[1]
    
    vectors = {}
    lengths = {}
    for i in range(1, 101):
        path = f"tf_idf_lemmas/{i}.txt"
        if os.path.exists(path):
            vectors[i] = {}
            sum_sq = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 3:
                        lemma, tfidf = p[0], float(p[2])
                        vectors[i][lemma] = tfidf
                        sum_sq += tfidf**2
            lengths[i] = math.sqrt(sum_sq) if sum_sq > 0 else 1
    return url_map, vectors, lengths

# --- ЭНДПОИНТЫ ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/search")
async def search(q: str = Query(None)):
    if not q:
        return []

    url_map, doc_vectors, doc_lengths = get_data()
    
    # Обработка запроса
    query_words = re.findall(r'\b[a-zA-Z]+\b', q.lower())
    query_lemmas = [lemmatizer.lemmatize(w) for w in query_words]
    
    query_vec = {}
    for l in query_lemmas:
        query_vec[l] = query_vec.get(l, 0) + 1
    
    q_len = math.sqrt(sum(v**2 for v in query_vec.values())) if query_vec else 1

    results = []
    for doc_id, doc_vec in doc_vectors.items():
        dot_product = sum(query_vec.get(l, 0) * doc_vec.get(l, 0) for l in query_lemmas)
        
        if dot_product > 0:
            score = dot_product / (q_len * doc_lengths[doc_id])
            
            # Берем TF-IDF первой леммы запроса для показа в UI
            main_lemma = query_lemmas[0]
            tfidf_val = doc_vec.get(main_lemma, 0)
            
            results.append({
                "doc_id": doc_id,
                "url": url_map.get(doc_id, f"Local Page {doc_id}"),
                "score": round(score, 4),
                "tfidf": round(tfidf_val, 6),
                "lemma": main_lemma
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# Дополнительный эндпоинт для запуска краулера (демо-версия)
@app.post("/api/crawl")
async def crawl_endpoint():
    # Здесь можно вызвать вашу функцию run_crawler()
    # Для безопасности в демо возвращаем статус
    return {"status": "Crawler started", "message": "100 pages are being processed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)