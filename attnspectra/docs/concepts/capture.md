# Captura de atención

Este documento explica qué es un `CapturedRun`, cómo se construye internamente y qué shapes tienen los tensores en cada paso del pipeline.

---

## ¿Qué es un `CapturedRun`?

Un `CapturedRun` es el objeto central del paquete. Encapsula **todo lo que se captura durante un único forward pass** de un modelo transformer:

```python
@dataclass
class CapturedRun:
    input_ids:   Tensor              # (B, T)        — ids de entrada
    attentions:  List[Tensor | None] # L × (B,H,Q,K) — atención post-softmax por capa
    scores:      List[Tensor | None] # L × (B,H,Q,K) — logits Q·K^T pre-softmax (opcional)
    token_strs:  List[str]           # T strings      — tokens decodificados
    model_info:  ModelInfo           # metadatos del modelo
    style_idx:   int | None          # índice de estilo (para modelos con style embedding)
```

Una vez que tienes un `CapturedRun`, el resto del paquete (`analysis/`, `viz/`, `storage/`) lo consume directamente sin necesidad de conocer el modelo origen.

---

## Shapes canónicos

| Campo | Shape | Descripción |
|-------|-------|-------------|
| `input_ids` | `(B, T)` | B = batch size, T = longitud de secuencia |
| `attentions[i]` | `(B, H, Q, K)` | H = n_heads, Q = K = T para atención causal |
| `scores[i]` | `(B, H, Q, K)` | Logits Q·K^T antes del softmax |
| `token_strs` | `[str] × T` | Decodificado para la primera secuencia (batch=0) |

En atención causal (decoder-only): Q = K = T, y la mitad superior de cada matriz `(Q, K)` está enmascarada con `-inf` antes del softmax.

Las capas no capturadas (por `target_layers`) tienen `None` en lugar de un tensor.

---

## Pipeline de captura

```
texto / ids
    │
    ▼
┌─────────────────────────────────┐
│           Adapter               │
│  (CustomGPTAdapter /            │
│   HFTransformerAdapter)         │
│                                 │
│  1. Tokenizar (si necesario)    │
│  2. Construir capture_dict      │
│  3. model.forward(capture=...)  │
│  4. _parse_caches()             │
└────────────┬────────────────────┘
             │  raw caches / raw attentions
             ▼
┌─────────────────────────────────┐
│       capture/normalize.py      │
│                                 │
│  normalize_cache_list()         │  ← GPT custom
│  normalize_hf_attentions()      │  ← HuggingFace
│                                 │
│  Convierte al formato canónico: │
│  List[Tensor(B,H,Q,K) | None]   │
└────────────┬────────────────────┘
             │
             ▼
        CapturedRun
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
analysis/  viz/      storage/
```

---

## Adapters

### `CustomGPTAdapter`

Para modelos con argumento `capture` en el `forward()`:

```python
adapter = CustomGPTAdapter(
    model=model,
    tokenizer=tokenizer,
    n_layers=8, n_heads=8, d_model=512,
)
run = adapter.capture_from_ids(input_ids, config, style_idx=0)
```

El `capture_dict` que se pasa al modelo se construye automáticamente a partir del `CaptureConfig`:

```python
capture_dict = {
    "attn_pre":  config.capture_attn_pre,   # True por defecto
    "attn_post": config.capture_attn_post,  # False por defecto
    "scores":    config.capture_scores,     # False por defecto
}
```

### `HFTransformerAdapter`

Para modelos HuggingFace (BERT, GPT-2, T5...):

```python
adapter = HFTransformerAdapter(model=model, tokenizer=tokenizer)
run = adapter.capture_from_text("texto de prueba", config)
```

Usa `output_attentions=True` internamente. Los modelos HF devuelven tensores ya en formato `(B, H, Q, K)`. `normalize_hf_attentions()` solo aplica el filtro de capas.

---

## `CaptureConfig`

Controla qué se captura:

```python
config = CaptureConfig(
    capture_attn_pre  = True,    # atención post-softmax (recomendado)
    capture_attn_post = False,   # atención post-dropout (más memoria)
    capture_scores    = False,   # logits Q·K^T pre-softmax — activa métricas Sc_*
    target_layers     = None,    # None = todas; [0,3,7] = capas específicas
    max_seq_len       = None,    # truncar secuencias largas
)
```

**`capture_attn_pre` vs `capture_attn_post`:**

```
scores (logits) ──softmax──▶ attn_pre ──dropout──▶ attn_post ──@ V──▶ output
                                 ↑                     ↑
                          capture_attn_pre        capture_attn_post
                          (recomendado)           (solo difiere en training)
```

En modo `eval()` (inferencia), `attn_pre == attn_post` porque el dropout está desactivado. Usa `capture_attn_pre` salvo que necesites específicamente el comportamiento de training.

**`capture_scores=True` — activa la familia Sc_*:**

Al activar `capture_scores`, los logits Q·K^T quedan disponibles en `run.scores`. Esto permite que `compute_head_metrics()` calcule las métricas de la familia `Sc_*` sobre los scores reales, sin la distorsión del softmax:

```python
config = CaptureConfig(capture_attn_pre=True, capture_scores=True)
run    = adapter.capture_from_ids(input_ids, config)
metrics = compute_head_metrics(run)

if metrics.has_scores_metrics():
    print(metrics.Sc_effective_rank)  # rango efectivo sin distorsión de softmax
```

**`target_layers`:**

```python
# Capturar solo las capas 0, 3 y 7
config = CaptureConfig(target_layers=[0, 3, 7])

# Capturar solo la última capa
config = CaptureConfig(target_layers=[-1])

# Capturar todas (por defecto)
config = CaptureConfig(target_layers=None)
```

Las capas no capturadas aparecen como `None` en `run.attentions` y `run.scores`.

---

## Acceso a los datos

```python
# Propiedades básicas
run.n_layers   # 8
run.seq_len    # 15
run.batch_size # 1

# Tokens decodificados (batch=0)
run.token_strs  # ['<bos>', '<wiki>', '<sep>', 'La', ' foto', ...]

# Matriz de atención de una cabeza concreta (batch=0)
A = run.get_attention(layer=3, head=2)  # (Q, K) = (T, T)

# Scores Q·K^T (solo si capture_scores=True)
Sc = run.get_scores(layer=3, head=2)   # (Q, K)
```

### Helpers de `capture/selectors.py`

```python
import attnspectra as aspec

# Submatriz sin tokens de prefijo, lista para plotear
A, tokens = aspec.get_content_attention(run, layer=3, head=2, prefix_len=3)

# Tokens más atendidos en una cabeza
top = aspec.most_attended_tokens(run, layer=5, head=3, topk=5)

# Extraer todas las cabezas de una capa
A_layer = aspec.get_layer_head(run, layer=3, head=0)  # (Q, K)

# Seleccionar tokens por índice
tokens_sel = aspec.pick_tokens(run, indices=[0, 3, 7])
```

---

## API de alto nivel — `capture/api.py`

Para flujos comunes, el módulo `capture/api.py` ofrece funciones orquestadoras que combinan adapter + config + métricas en una sola llamada:

```python
# Captura y métricas en una llamada
run, metrics = aspec.capture_and_compute(adapter, input_ids, style_idx=0)

# Procesar una lista de secuencias
runs, metrics_list = aspec.capture_batch(adapter, ids_list, style_idx=0)

# Experimento de degradación completo (EXP1)
per_cond = aspec.run_degradation_exp(
    adapter, base_ids, vocab_size=32000, prefix_len=3, style_idx=0
)

# Comparación de estilos (EXP3)
metrics_a, metrics_b = aspec.run_style_comparison(
    adapter, ids_list, style_a=0, style_b=1
)
```

---

## Normalización interna

Antes de calcular métricas, `compute_head_metrics` aplica `normalize_attn(A)` para garantizar que las filas sumen 1, incluso si el modelo devolvió matrices con pequeñas imprecisiones numéricas:

```python
A_norm = normalize_attn(A)   # ∑ⱼ A_norm[i,j] = 1.0 para todo i
```

`normalize_attn` reemplaza NaN e Inf antes de dividir, por lo que es segura incluso con matrices de atención causal donde las posiciones enmascaradas pueden tener valores extremos.

---

## Serialización

```python
# Guardar
save_run(run, "run.npz")
save_metrics(metrics, "metrics.json")

# Cargar
run2     = load_run("run.npz")
metrics2 = load_metrics("metrics.json")
```

El fichero `.npz` contiene:
- `input_ids`: array numpy `(B, T)`
- `attentions_L000`, `attentions_L001`, ...: un array por capa no nula
- `_meta`: JSON con `model_info`, `token_strs`, `style_idx` y `schema_version`

> **Nota:** los scores no se serializan para reducir el tamaño del fichero.

`load_metrics()` es retrocompatible: campos añadidos en versiones posteriores (como los campos `Sc_*` de v0.2.0) se rellenan con NaN al cargar ficheros antiguos.

---

## Diagrama de shapes completo

```
input_ids: (B=1, T=15)
                │
                │  forward pass
                ▼
attentions: [
  L0: (1, 8, 15, 15),   ← capa 0, 8 cabezas, 15 queries × 15 keys
  L1: (1, 8, 15, 15),
  L2: None,              ← no capturada (target_layers=[0,1,3,...])
  L3: (1, 8, 15, 15),
  ...
]
scores: [               ← solo si capture_scores=True
  L0: (1, 8, 15, 15),
  L1: (1, 8, 15, 15),
  L2: None,
  L3: (1, 8, 15, 15),
  ...
]

run.get_attention(layer=3, head=2)
  → A:  (15, 15)   ← batch=0, head=2, post-softmax ∈ [0,1]

run.get_scores(layer=3, head=2)
  → Sc: (15, 15)   ← batch=0, head=2, pre-softmax ∈ (-∞, +∞)

compute_head_metrics(run)
  → metrics.A_attn_entropy:        (8, 15)  ← familia A, siempre disponible
    metrics.A_effective_rank:       (8, 15)
    metrics.A_eigenvalue_decay:     (8, 15)
    metrics.A_spectral_decay_rate:  (8, 15)
    metrics.A_anisotropy_index:     (8, 15)
    metrics.A_gini:                 (8, 15)
    metrics.A_attn_distance:        (8, 15)
    metrics.S_effective_rank:       (8, 15)  ← familia S, siempre disponible
    metrics.Sc_effective_rank:      (8, 15)  ← familia Sc, solo con capture_scores=True
    metrics.Sc_eigenvalue_decay:    (8, 15)
    ...
```