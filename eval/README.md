# HEP Embedding Evaluation Pipeline

Evaluación de modelos de embeddings para **búsqueda semántica de papers de física de altas energías (HEP)**.

---

## Estructura del proyecto

```
hep_eval/
├── 01_prepare_data.py        # Descarga papers de InspireHEP y construye ground-truth
├── 02_generate_embeddings.py # Genera embeddings con el modelo elegido
├── 03_evaluate.py            # Calcula métricas y genera reporte comparativo
├── requirements.txt
│
├── data/                     # (generado) corpus, queries, qrels
├── embeddings/               # (generado) embeddings por modelo
└── results/                  # (generado) métricas y gráficos
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso paso a paso

### Paso 1 — Preparar datos

Descarga papers de InspireHEP (hep-ph, hep-ex, hep-th, hep-lat) y construye
el ground-truth de relevancia por co-citación y categoría.

```bash
python 01_prepare_data.py \
    --papers-per-category 300 \   # papers por categoría HEP (~1200 total)
    --n-queries 200 \             # queries de evaluación
    --output-dir data
```

Genera:
- `data/corpus.jsonl`   → todos los papers (id, title, abstract, category)
- `data/queries.jsonl`  → subset de papers usados como queries
- `data/qrels.jsonl`    → pares (query_id, doc_id, score) de relevancia

**Niveles de relevancia:**
| Score | Significado |
|-------|-------------|
| 2     | El paper corpus es citado por la query (altamente relevante) |
| 1     | Mismo subcampo HEP (relevante) |

---

### Paso 2 — Generar embeddings

```bash
# SPECTER 2 (recomendado para retrieval científico)
python 02_generate_embeddings.py --model specter2

# SciNCL (entrenado con señal de co-citación)
python 02_generate_embeddings.py --model scincl

# Baseline general para comparación
python 02_generate_embeddings.py --model minilm
```

**Alias de modelos disponibles:**

| Alias     | Modelo HuggingFace                              |
|-----------|-------------------------------------------------|
| specter2  | allenai/specter2_base                           |
| specter1  | allenai/specter                                 |
| scincl    | malteos/scincl                                  |
| minilm    | sentence-transformers/all-MiniLM-L6-v2          |
| scibert   | allenai/scibert_scivocab_uncased                |

También puedes pasar un nombre completo de HuggingFace:
```bash
python 02_generate_embeddings.py --model "allenai/specter2_base"
```

---

### Paso 3 — Evaluar

```bash
# Evaluar todos los modelos con embeddings generados
python 03_evaluate.py

# Evaluar un modelo específico
python 03_evaluate.py --model specter2

# Cambiar los valores de k
python 03_evaluate.py --k-values 1 5 10 50 100
```

Genera:
- `results/report.json`          → métricas completas por modelo (JSON)
- `results/comparison.csv`       → tabla comparativa
- `results/comparison_plot.png`  → gráfico de barras comparativo

---

## Métricas calculadas

| Métrica       | Descripción |
|---------------|-------------|
| **NDCG@k**    | Calidad del ranking considerando posición y score de relevancia |
| **MRR**       | Rango reciproco medio del primer relevante recuperado |
| **Recall@k**  | Fracción de relevantes recuperados en top-k |
| **Precision@k**| Fracción del top-k que es relevante |
| **MAP**       | Precisión media interpolada sobre todos los relevantes |

---

## Ejemplo de salida esperada

```
──────────────────────────────────────────────────────────────────
Modelo                          ndcg@10    ndcg@100         mrr  recall@10  recall@100        map
──────────────────────────────────────────────────────────────────
specter2                         0.4231      0.3876      0.5102     0.2841      0.6234     0.3127
scincl                           0.3987      0.3712      0.4876     0.2654      0.5981     0.2934
specter1                         0.3654      0.3401      0.4521     0.2401      0.5712     0.2687
minilm                           0.2213      0.2104      0.2987     0.1532      0.3901     0.1842
──────────────────────────────────────────────────────────────────
```

---

## Notas

- La primera ejecución de `02_generate_embeddings.py` descarga los pesos del modelo (~400 MB para SPECTER2).
- Con GPU, la generación de embeddings es ~10× más rápida.
- El ground-truth por co-citación es ruidoso pero válido para evaluación comparativa; para evaluaciones de publicación se recomienda anotación humana adicional.
