"""
02_generate_embeddings.py
=========================
Genera embeddings para el corpus y las queries usando modelos especializados en HEP/ciencia.

Modelos soportados:
    - allenai/specter2_base          (SPECTER 2 — recomendado para retrieval)
    - allenai/specter                (SPECTER 1 — baseline clásico)
    - malteos/scincl                 (SciNCL — entrenado con co-citación)
    - sentence-transformers/all-MiniLM-L6-v2  (baseline general, para comparar)
    - Cualquier modelo HuggingFace compatible con sentence-transformers

Salida:
    embeddings/<model_slug>/corpus_embeddings.npy
    embeddings/<model_slug>/query_embeddings.npy
    embeddings/<model_slug>/corpus_ids.json
    embeddings/<model_slug>/query_ids.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Modelos disponibles ────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "specter2":    "allenai/specter2_base",
    "specter1":    "allenai/specter",
    "scincl":      "malteos/scincl",
    "minilm":      "sentence-transformers/all-MiniLM-L6-v2",
    "scibert":     "allenai/scibert_scivocab_uncased",
}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_text(paper: dict, model_name: str) -> str:
    """
    Formatea el texto de entrada según las convenciones de cada modelo.
    SPECTER usa el separador [SEP] entre título y abstract.
    """
    title    = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()

    if "specter" in model_name.lower():
        # Formato oficial SPECTER: title + [SEP] + abstract
        return f"{title} [SEP] {abstract}"
    else:
        # Formato estándar para sentence-transformers
        return f"{title}. {abstract}" if title else abstract


def get_embedder(model_name: str):
    """
    Devuelve un objeto con método .encode() compatible.
    Detecta automáticamente si usar sentence-transformers o HuggingFace directo.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Cargando '{model_name}' con sentence-transformers...")
        model = SentenceTransformer(model_name)
        return model, "sbert"
    except Exception:
        pass

    # Fallback: HuggingFace Transformers con mean pooling manual
    from transformers import AutoTokenizer, AutoModel
    import torch

    print(f"  Cargando '{model_name}' con HuggingFace Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    return (tokenizer, model), "hf"


def encode_texts(texts: list[str], embedder, mode: str,
                 batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Genera embeddings en batches."""
    import torch

    if mode == "sbert":
        embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=device,
        )
        return np.array(embeddings)

    # Modo HuggingFace manual
    tokenizer, model = embedder
    model.to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**encoded)
        # CLS token como representación
        emb = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-9)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    data_dir = Path(args.data_dir)
    emb_dir  = Path(args.embeddings_dir)

    # Resolver alias de modelo
    model_name = SUPPORTED_MODELS.get(args.model, args.model)
    model_slug = args.model.replace("/", "_")
    out_dir    = emb_dir / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Modelo: {model_name}")
    print(f"   Guardando en: {out_dir}/\n")

    # Detectar device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" \
                 if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"   Device: {device}")

    # Cargar datos
    corpus  = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    print(f"   Corpus:  {len(corpus)} docs")
    print(f"   Queries: {len(queries)} queries\n")

    # Cargar modelo
    embedder, mode = get_embedder(model_name)

    # Textos formateados
    corpus_texts = [format_text(p, model_name) for p in corpus]
    query_texts  = [format_text(p, model_name) for p in queries]

    # Generar embeddings
    print("🔢 Generando embeddings del corpus...")
    corpus_emb = encode_texts(corpus_texts, embedder, mode,
                              batch_size=args.batch_size, device=device)

    print("\n🔢 Generando embeddings de las queries...")
    query_emb  = encode_texts(query_texts, embedder, mode,
                              batch_size=args.batch_size, device=device)

    # Guardar
    np.save(out_dir / "corpus_embeddings.npy", corpus_emb)
    np.save(out_dir / "query_embeddings.npy",  query_emb)

    with open(out_dir / "corpus_ids.json", "w") as f:
        json.dump([p["id"] for p in corpus], f)
    with open(out_dir / "query_ids.json", "w") as f:
        json.dump([p["id"] for p in queries], f)

    print(f"\n✅ Embeddings guardados")
    print(f"   corpus_embeddings.npy → shape {corpus_emb.shape}")
    print(f"   query_embeddings.npy  → shape {query_emb.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera embeddings para papers HEP")
    parser.add_argument("--model",          default="specter2",
                        help=f"Modelo a usar. Alias: {list(SUPPORTED_MODELS.keys())} "
                             f"o cualquier nombre HuggingFace.")
    parser.add_argument("--data-dir",       default="data")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--batch-size",     type=int, default=32)
    main(parser.parse_args())
