"""
01_prepare_data.py
==================
Descarga y prepara un dataset de evaluación de papers HEP desde la API de InspireHEP.

Salida:
    data/corpus.jsonl       → Todos los papers (id, title, abstract)
    data/queries.jsonl      → Papers de consulta (subconjunto)
    data/qrels.jsonl        → Relevancia ground-truth basada en co-citación / categoría
"""

import json
import time
import random
import argparse
import requests
from pathlib import Path
from collections import defaultdict

# ── Configuración ─────────────────────────────────────────────────────────────

INSPIREHEP_API = "https://inspirehep.net/api/literature"

# Categorías HEP de interés (arxiv)
HEP_CATEGORIES = [
    "hep-ph",   # Phenomenology
    "hep-ex",   # Experiment
    "hep-th",   # Theory
    "hep-lat",  # Lattice
]

# ── Funciones de descarga ──────────────────────────────────────────────────────

def fetch_papers(category: str, max_papers: int = 500, page_size: int = 25) -> list[dict]:
    """Descarga papers de InspireHEP filtrando por categoría arxiv."""
    papers = []
    page = 1

    print(f"  Descargando '{category}'...")
    while len(papers) < max_papers:
        params = {
            "sort": "mostrecent",
            "size": page_size,
            "page": page,
            "fields": "arxiv_eprints,titles,abstracts,inspire_categories,references",
            "q": f"arxiv_categories {category} and a abstracts",
        }
        try:
            resp = requests.get(INSPIREHEP_API, params=params, timeout=30)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                meta = hit.get("metadata", {})
                abstract = (meta.get("abstracts") or [{}])[0].get("value", "").strip()
                title    = (meta.get("titles")    or [{}])[0].get("title",  "").strip()
                inspire_id = str(hit.get("id", ""))
                refs = [
                    str(r.get("record", {}).get("$ref", "").split("/")[-1])
                    for r in (meta.get("references") or [])
                    if r.get("record")
                ]
                if abstract and title and inspire_id:
                    papers.append({
                        "id":       inspire_id,
                        "title":    title,
                        "abstract": abstract,
                        "category": category,
                        "refs":     refs,
                    })
            print(f"    página {page}: {len(papers)} papers acumulados")
            page += 1
            time.sleep(0.5)   # respetar rate-limit
        except Exception as e:
            print(f"    Error en página {page}: {e}")
            break

    return papers[:max_papers]


def build_qrels_from_references(papers: list[dict]) -> dict[str, dict[str, int]]:
    """
    Construye relevancia ground-truth: un paper es relevante para una query
    si es citado por ella (relevance=2) o comparten categoría (relevance=1).
    """
    id_set  = {p["id"] for p in papers}
    id2cat  = {p["id"]: p["category"] for p in papers}
    qrels   = defaultdict(dict)

    for paper in papers:
        qid = paper["id"]
        # Relevancia 2: citado directamente
        for ref_id in paper["refs"]:
            if ref_id in id_set and ref_id != qid:
                qrels[qid][ref_id] = 2
        # Relevancia 1: misma categoría (si no ya marcado)
        for other in papers:
            oid = other["id"]
            if oid != qid and oid not in qrels[qid]:
                if id2cat[oid] == paper["category"]:
                    qrels[qid][oid] = 1

    # Filtrar queries sin ningún relevante con score 2
    qrels = {qid: rels for qid, rels in qrels.items()
             if any(s == 2 for s in rels.values())}
    return qrels


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Descargar papers
    all_papers = []
    per_cat = args.papers_per_category
    for cat in HEP_CATEGORIES:
        all_papers.extend(fetch_papers(cat, max_papers=per_cat))

    # Deduplicar por id
    seen = set()
    corpus = []
    for p in all_papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            corpus.append(p)
    print(f"\nCorpus total: {len(corpus)} papers únicos")

    # 2. Ground-truth
    print("Construyendo qrels...")
    qrels = build_qrels_from_references(corpus)
    query_ids = set(qrels.keys())
    print(f"Queries con relevantes citados: {len(query_ids)}")

    # 3. Muestrear queries (no queremos evaluar todos)
    random.seed(42)
    sampled_query_ids = set(random.sample(sorted(query_ids),
                                          min(args.n_queries, len(query_ids))))

    # 4. Guardar
    # corpus.jsonl
    with open(out_dir / "corpus.jsonl", "w") as f:
        for p in corpus:
            f.write(json.dumps({"id": p["id"], "title": p["title"],
                                "abstract": p["abstract"], "category": p["category"]},
                               ensure_ascii=False) + "\n")

    # queries.jsonl
    queries = [p for p in corpus if p["id"] in sampled_query_ids]
    with open(out_dir / "queries.jsonl", "w") as f:
        for p in queries:
            f.write(json.dumps({"id": p["id"], "title": p["title"],
                                "abstract": p["abstract"]},
                               ensure_ascii=False) + "\n")

    # qrels.jsonl  (formato BEIR: query_id, doc_id, score)
    with open(out_dir / "qrels.jsonl", "w") as f:
        for qid, rels in qrels.items():
            if qid in sampled_query_ids:
                for doc_id, score in rels.items():
                    f.write(json.dumps({"query_id": qid,
                                        "doc_id": doc_id,
                                        "score": score}) + "\n")

    print(f"\n✅ Archivos guardados en '{out_dir}/'")
    print(f"   corpus.jsonl  → {len(corpus)} docs")
    print(f"   queries.jsonl → {len(queries)} queries")
    print(f"   qrels.jsonl   → {sum(len(v) for v in qrels.values() if any(qid in sampled_query_ids for qid in [list(qrels.keys())[0]]))} pares relevantes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset HEP para evaluación")
    parser.add_argument("--output-dir",            default="data",  help="Directorio de salida")
    parser.add_argument("--papers-per-category",   type=int, default=300, help="Papers por categoría HEP")
    parser.add_argument("--n-queries",             type=int, default=200, help="Nº de queries de evaluación")
    main(parser.parse_args())
