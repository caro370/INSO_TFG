# Guía de inicio rápido

Esta guía muestra cómo usar `attnspectra` desde cero en 5 pasos.

---

## Paso 1 — Instalación

```bash
cd attnspectra/
pip install -e ".[dev]"        # instalación básica + herramientas de desarrollo
pip install -e ".[dev,hf]"     # añade soporte HuggingFace (BERT, GPT-2, T5...)
```

Verificar que funciona:

```bash
python -c "import attnspectra; print(attnspectra.__version__)"

# 0.1.0
```

---

## Paso 2 — Conectar tu modelo

### Opción A: modelo custom con mecanismo `capture`

```python
import attnspectra as aspec

adapter = aspec.CustomGPTAdapter(
    model=model,
    tokenizer=tokenizer,
    n_layers=8,
    n_heads=8,
    d_model=512,
    model_name="mi-gpt",
)
```

### Opción B: modelo HuggingFace

```python
from transformers import AutoModel, AutoTokenizer

model     = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

adapter = aspec.HFTransformerAdapter(
    model=model,
    tokenizer=tokenizer,
    model_name="bert-base-uncased",
)
```

---

## Paso 3 — Capturar atención

```python
import torch

# Solo atención post-softmax (siempre disponible)
config = aspec.CaptureConfig(
    capture_attn_pre=True,
    capture_scores=False,   # True para activar las métricas Sc_*
    target_layers=None,     # None = todas las capas
)

# Con modelo custom (ids construidos manualmente)
ids = [bos_id, wiki_tok_id, sep_id] + tokenizer.encode(texto).ids
input_ids = torch.tensor([ids])
run = adapter.capture_from_ids(input_ids, config, style_idx=0)

# Con HuggingFace (el adapter tokeniza internamente)
run = adapter.capture_from_text("Texto de prueba", config)

print(f"Capturado: {run.n_layers} capas, {run.seq_len} tokens")
print(f"Tokens: {run.token_strs}")
```

O usa la API de alto nivel que combina captura y métricas en una sola llamada:

```python
run, metrics = aspec.capture_and_compute(adapter, input_ids, style_idx=0)
```

---

## Paso 4 — Calcular métricas

```python
metrics = aspec.compute_head_metrics(run)

# Las 14 métricas tienen shape (n_layers, n_heads)
print(metrics.A_attn_entropy.shape)       # (8, 8)
print(metrics.A_effective_rank[3])        # rango efectivo de la capa 3, todas las cabezas
print(metrics.A_attn_distance)            # distancia media de atención por capa × cabeza

# Parámetros opcionales
metrics = aspec.compute_head_metrics(
    run,
    causal=True,          # True para GPT (decoder-only), False para BERT (encoder)
)
```

Las métricas están organizadas en tres familias:

| Prefijo | Matriz | Disponible |
|---------|--------|------------|
| `A_*`  | post-softmax A ∈ [0,1] | siempre |
| `S_*`  | log(A), aproxima scores | siempre |
| `Sc_*` | scores Q·K^T reales | solo con `capture_scores=True` |

Para activar las métricas `Sc_*`:

```python
config  = aspec.CaptureConfig(capture_attn_pre=True, capture_scores=True)
run, metrics = aspec.capture_and_compute(adapter, input_ids)

if metrics.has_scores_metrics():
    print(metrics.Sc_effective_rank)   # rango efectivo sin distorsión de softmax
```

Ver `docs/concepts/metrics.md` para la definición completa de cada métrica.

---

## Paso 5 — Visualizar y guardar

```python
# Matriz de atención de una cabeza (sin tokens de prefijo)
A, tokens = aspec.get_content_attention(run, layer=3, head=2, prefix_len=3)
fig = aspec.plot_attention_matrix(A, tokens, title="L3·H2")
fig.savefig("attn_matrix.png")

# Heatmap de métricas por capa × cabeza
fig2 = aspec.heatmap_metric(metrics.A_attn_entropy, title="Entropía por capa y cabeza")
fig2.savefig("entropy_heatmap.png")

# Distancia de atención: ¿capas bajas locales, capas altas globales?
fig3 = aspec.heatmap_metric(metrics.A_attn_distance, title="Distancia media de atención")
fig3.savefig("attn_distance.png")

# Guardar para análisis posterior
aspec.save_run(run, "run.npz")
aspec.save_metrics(metrics, "metrics.json")

# Cargar más tarde (retrocompatible con versiones anteriores)
run2     = aspec.load_run("run.npz")
metrics2 = aspec.load_metrics("metrics.json")
```

---

## Acceso rápido a datos

```python
# Tokens más atendidos en una cabeza
top = aspec.most_attended_tokens(run, layer=5, head=3, topk=5)

# Extraer una métrica por capa o por cabeza
entropy_capa5  = aspec.get_metric_layer(metrics, "A_attn_entropy", layer=5)  # (n_heads,)
entropy_cab2   = aspec.get_metric_head(metrics, "A_attn_entropy", head=2)    # (n_layers,)

# Cabezas más sensibles a una diferencia entre condiciones
heads = aspec.top_sensitive_heads(metrics_wiki, metrics_poem,
                                   key="A_attn_entropy", topk=10)
for layer, head, delta, val_a, val_b in heads:
    print(f"L{layer} H{head}: |Δ|={delta:.4f}  (wiki={val_a:.3f}, poem={val_b:.3f})")
```

---

## Experimento de degradación (EXP1)

```python
aspec.set_seed(42)

# Opción A: usando la API de alto nivel (recomendado)
per_cond = aspec.run_degradation_exp(
    adapter, base_ids,
    vocab_size=32000,
    prefix_len=3,
    style_idx=0,
)

# Opción B: manual con make_variants
variants = aspec.make_variants(base_ids, vocab_size=32000, prefix_len=3)
# {"clean", "shuffle", "replace_10%", "replace_30%", "replace_50%", "random_iid"}
per_cond = {}
for name, vid in variants.items():
    vrun = adapter.capture_from_ids(torch.tensor([vid]), config, style_idx=0)
    per_cond[name] = [aspec.compute_head_metrics(vrun)]

# Visualizar Δ entropía vs capa
fig = aspec.plot_delta_lines_by_condition(per_cond, metric_key="A_attn_entropy")
fig.savefig("degradacion.png")
```

---

## Comparación de estilos (EXP3)

```python
# Opción A: API de alto nivel
metrics_wiki, metrics_poem = aspec.run_style_comparison(
    adapter, ids_list, style_a=0, style_b=1
)

# Heatmap de diferencia entre estilos
fig = aspec.heatmap_delta(
    metrics_wiki[0].A_effective_rank,
    metrics_poem[0].A_effective_rank,
    title="Δ rango efectivo (wiki - poem)",
)
fig.savefig("delta_er.png")
```

---

## Ejecutar los tests

```bash
pytest tests/ -v
# 97 tests en total
```

---