"""
03_evaluate.py
==============
Evalúa modelos de embeddings para búsqueda semántica de papers HEP.

Métricas calculadas:
    - NDCG@k        (Normalized Discounted Cumulative Gain)
    - MRR           (Mean Reciprocal Rank)
    - Recall@k
    - Precision@k
    - MAP           (Mean Average Precision)

Salida:
    results/report.json        → Métricas por modelo
    results/comparison.csv     → Tabla comparativa
    results/comparison_plot.png → Gráfico comparativo (si matplotlib disponible)
"""

import json
import argparse
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict
from typing import Optional


# ── Métricas de Retrieval ──────────────────────────────────────────────────────

def dcg_at_k(scores: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    scores = scores[:k]
    return sum(s / np.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """NDCG@k dado el ranking recuperado y los relevantes con sus scores."""
    if not relevant:
        return 0.0
    scores   = [relevant.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    ideal    = sorted(relevant.values(), reverse=True)
    return dcg_at_k(scores, k) / (dcg_at_k(ideal, k) or 1)


def reciprocal_rank(retrieved_ids: list[str], relevant: dict[str, int],
                    min_score: int = 1) -> float:
    """Mean Reciprocal Rank — primer relevante en el ranking."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if relevant.get(doc_id, 0) >= min_score:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant: dict[str, int],
                k: int, min_score: int = 1) -> float:
    n_relevant = sum(1 for s in relevant.values() if s >= min_score)
    if n_relevant == 0:
        return 0.0
    n_hit = sum(1 for doc_id in retrieved_ids[:k]
                if relevant.get(doc_id, 0) >= min_score)
    return n_hit / n_relevant


def precision_at_k(retrieved_ids: list[str], relevant: dict[str, int],
                   k: int, min_score: int = 1) -> float:
    if k == 0:
        return 0.0
    n_hit = sum(1 for doc_id in retrieved_ids[:k]
                if relevant.get(doc_id, 0) >= min_score)
    return n_hit / k


def average_precision(retrieved_ids: list[str], relevant: dict[str, int],
                      min_score: int = 1) -> float:
    n_relevant = sum(1 for s in relevant.values() if s >= min_score)
    if n_relevant == 0:
        return 0.0
    hits, score = 0, 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if relevant.get(doc_id, 0) >= min_score:
            hits  += 1
            score += hits / rank
    return score / n_relevant


# ── Búsqueda por similitud coseno ─────────────────────────────────────────────

def cosine_retrieval(query_emb: np.ndarray, corpus_emb: np.ndarray,
                     corpus_ids: list[str], top_k: int = 100) -> list[str]:
    """Devuelve los top_k corpus_ids más similares a query_emb."""
    # Los embeddings ya están L2-normalizados → producto punto = coseno
    sims    = corpus_emb @ query_emb          # (N,)
    indices = np.argpartition(sims, -top_k)[-top_k:]
    indices = indices[np.argsort(sims[indices])[::-1]]
    return [corpus_ids[i] for i in indices]


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                qrels[r["query_id"]][r["doc_id"]] = r["score"]
    return dict(qrels)


def load_embeddings(model_dir: Path):
    corpus_emb = np.load(model_dir / "corpus_embeddings.npy")
    query_emb  = np.load(model_dir / "query_embeddings.npy")
    with open(model_dir / "corpus_ids.json") as f:
        corpus_ids = json.load(f)
    with open(model_dir / "query_ids.json") as f:
        query_ids  = json.load(f)
    return corpus_emb, query_emb, corpus_ids, query_ids


# ── Evaluación de un modelo ────────────────────────────────────────────────────

def evaluate_model(model_dir: Path, qrels: dict, ks: list[int] = [1, 5, 10, 20, 100],
                   top_k: int = 100) -> dict:
    """Evalúa un modelo y devuelve sus métricas agregadas."""
    corpus_emb, query_emb, corpus_ids, query_ids = load_embeddings(model_dir)

    results = defaultdict(list)

    for i, qid in enumerate(query_ids):
        if qid not in qrels:
            continue

        relevant = qrels[qid]
        q_emb    = query_emb[i]

        # Excluir la query del corpus (si está)
        mask        = np.ones(len(corpus_ids), dtype=bool)
        if qid in corpus_ids:
            mask[corpus_ids.index(qid)] = False
        filtered_corpus  = corpus_emb[mask]
        filtered_ids     = [cid for cid, m in zip(corpus_ids, mask) if m]

        retrieved = cosine_retrieval(q_emb, filtered_corpus, filtered_ids, top_k=top_k)

        results["mrr"].append(reciprocal_rank(retrieved, relevant))
        results["map"].append(average_precision(retrieved, relevant))
        for k in ks:
            results[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))
            results[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            results[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))

    # Promediar
    return {metric: float(np.mean(vals)) for metric, vals in results.items()}


# ── Guardar resultados ─────────────────────────────────────────────────────────

def save_report(all_results: dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    with open(out_dir / "report.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV comparativo
    if not all_results:
        return
    metrics = sorted(next(iter(all_results.values())).keys())
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + metrics)
        for model_name, res in all_results.items():
            writer.writerow([model_name] + [f"{res.get(m, 0):.4f}" for m in metrics])

    print(f"\n📊 Resultados guardados en '{out_dir}/'")
    _print_table(all_results, metrics)

    # Gráfico
    _plot_comparison(all_results, out_dir)


def _print_table(all_results: dict, metrics: list):
    """Imprime tabla comparativa en consola."""
    key_metrics = [m for m in ["ndcg@10", "ndcg@100", "mrr", "recall@10",
                                "recall@100", "map"] if m in metrics]
    header = f"{'Modelo':<30} " + "  ".join(f"{m:>12}" for m in key_metrics)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for model_name, res in all_results.items():
        row = f"{model_name:<30} " + "  ".join(
            f"{res.get(m, 0):>12.4f}" for m in key_metrics)
        print(row)
    print("─" * len(header))


def _plot_comparison(all_results: dict, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        key_metrics = ["ndcg@10", "ndcg@100", "mrr", "recall@10", "recall@100", "map"]
        key_metrics = [m for m in key_metrics
                       if m in next(iter(all_results.values()))]

        models  = list(all_results.keys())
        x       = np.arange(len(key_metrics))
        width   = 0.8 / max(len(models), 1)
        colors  = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, model in enumerate(models):
            vals = [all_results[model].get(m, 0) for m in key_metrics]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model,
                          color=colors[i % len(colors)], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(key_metrics, fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Comparativa de Modelos de Embeddings — HEP Retrieval", fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_plot.png", dpi=150)
        print(f"   Gráfico guardado: {out_dir}/comparison_plot.png")
    except ImportError:
        print("   (matplotlib no disponible, omitiendo gráfico)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    data_dir    = Path(args.data_dir)
    emb_dir     = Path(args.embeddings_dir)
    results_dir = Path(args.results_dir)

    # Cargar ground-truth
    qrels_path = data_dir / "qrels.jsonl"
    if not qrels_path.exists():
        raise FileNotFoundError(f"No encontrado: {qrels_path}")
    qrels = load_qrels(qrels_path)
    print(f"📋 Qrels cargados: {len(qrels)} queries con relevantes")

    # Detectar modelos disponibles (subdirectorios de embeddings/)
    model_dirs = sorted([d for d in emb_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        raise RuntimeError(f"No se encontraron embeddings en '{emb_dir}'. "
                           "Ejecuta primero 02_generate_embeddings.py.")

    # Si se especifica un modelo concreto, filtrar
    if args.model:
        model_dirs = [d for d in model_dirs if args.model in d.name]

    print(f"🔍 Evaluando {len(model_dirs)} modelo(s):\n"
          + "\n".join(f"   - {d.name}" for d in model_dirs))

    all_results = {}
    for model_dir in model_dirs:
        print(f"\n⏳ Evaluando: {model_dir.name} ...")
        try:
            metrics = evaluate_model(
                model_dir, qrels,
                ks=args.k_values,
                top_k=max(args.k_values),
            )
            all_results[model_dir.name] = metrics
            print(f"   NDCG@10={metrics.get('ndcg@10', 0):.4f}  "
                  f"MRR={metrics.get('mrr', 0):.4f}  "
                  f"Recall@100={metrics.get('recall@100', 0):.4f}")
        except Exception as e:
            print(f"   ⚠️  Error evaluando {model_dir.name}: {e}")

    save_report(all_results, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa embeddings para HEP retrieval")
    parser.add_argument("--data-dir",        default="data")
    parser.add_argument("--embeddings-dir",  default="embeddings")
    parser.add_argument("--results-dir",     default="results")
    parser.add_argument("--model",           default=None,
                        help="Evaluar solo este modelo (slug del directorio)")
    parser.add_argument("--k-values",        nargs="+", type=int,
                        default=[1, 5, 10, 20, 100],
                        help="Valores de k para métricas @k")
    main(parser.parse_args())
