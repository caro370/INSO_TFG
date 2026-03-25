# attnspectra

**attnspectra** es un paquete Python para extraer, analizar espectralmente y visualizar matrices de atención de modelos transformer.

---

## Instalación

```bash
git clone https://github.com/caro370/INSO_TFG.git
cd attnspectra

# Instalación básica + herramientas de desarrollo
pip install -e ".[dev]"

# Con soporte para modelos HuggingFace
pip install -e ".[dev,hf]"
```

**Requisitos:** Python ≥ 3.10, PyTorch ≥ 2.1, NumPy ≥ 1.24, Matplotlib ≥ 3.7

---

## Uso rápido

```python
import torch
import attnspectra as aspec

# 1. Crear el adapter para tu modelo
adapter = aspec.CustomGPTAdapter(
    model=model, tokenizer=tokenizer,
    n_layers=8, n_heads=8, d_model=512,
)

# 2. Capturar atención y calcular las 14 métricas en una sola llamada
config = aspec.CaptureConfig(capture_attn_pre=True)
run, metrics = aspec.capture_and_compute(adapter, input_ids, style_idx=0)

# 3. Acceder a las métricas — shape (n_layers, n_heads)
print(metrics.A_attn_entropy)        # entropía de atención
print(metrics.A_effective_rank)      # rango efectivo via SVD
print(metrics.A_attn_distance)       # distancia media de atención (tokens)
print(metrics.A_spectral_decay_rate) # velocidad de caída espectral

# 4. Visualizar
A, tokens = aspec.get_content_attention(run, layer=3, head=2, prefix_len=3)
fig = aspec.plot_attention_matrix(A, tokens)
fig2 = aspec.heatmap_metric(metrics.A_attn_distance, title="Distancia de atención")

# 6. Guardar resultados
aspec.save_run(run, "run.npz")
aspec.save_metrics(metrics, "metrics.json")
```

---

## Métricas disponibles

`compute_head_metrics()` calcula **14 métricas** organizadas en tres familias:

| Familia | Matriz analizada | Disponible | Nº métricas |
|---------|-----------------|------------|-------------|
| `A_*` | atención post-softmax A ∈ [0,1] | siempre | 7 |
| `S_*` | log(A), aproxima scores pre-softmax | siempre | 2 |
| `Sc_*` | scores Q·K^T reales | con `capture_scores=True` | 5 |

**Familia A** (7 métricas):
`A_attn_entropy`, `A_effective_rank`, `A_top_singular`,
`A_spectral_decay_rate`, `A_anisotropy_index`,
`A_gini`, `A_attn_distance`

**Familia S** (2 métricas):
`S_effective_rank`, `S_top_singular`

**Familia Sc** (5 métricas, requiere `capture_scores=True`):
`Sc_effective_rank`,  `Sc_top_singular`,
`Sc_spectral_decay_rate`, `Sc_anisotropy_index`, `Sc_gini`

Ver `docs/concepts/metrics.md` para la definición matemática e interpretación de cada métrica.

---

## Visualizaciones disponibles (`viz/`)

| Función | Descripción |
|---------|-------------|
| `plot_attention_matrix(A, tokens)` | Heatmap (T×T) de una cabeza con etiquetas de tokens |
| `plot_attention_matrix_interactive(run)` | Heatmap (T×T) con sliders de capa y cabeza| 
| `heatmap_metric(M)` | Heatmap (L×H) de una métrica absoluta |
| `heatmap_delta(A, B)` | Heatmap (L×H) de la diferencia A−B (colormap divergente) |
| `plot_metric_by_layer(metrics_list, key)` | Media ± std de una métrica vs capa |
| `plot_delta_lines_by_condition(per_cond, key, base)` | Δ vs capa por condición de degradación |

Todas las funciones devuelven `matplotlib.Figure` y aceptan `ax` opcional para incrustarlas en figuras compuestas.

---

## Estructura del paquete

```
src/attnspectra/
├── config/          — CaptureConfig, MetricConfig, PlotConfig
├── core/            — CapturedRun, HeadMetrics, ModelInfo, device, seeds
├── adapters/        — BaseAdapter, CustomGPTAdapter, HFTransformerAdapter
├── capture/         — normalize, selectors, api (orquestadores)
├── transforms/      — Perturbaciones de tokens (degradación)
├── analysis/        — 14 métricas espectrales y de entropía
├── viz/             — 6 funciones de visualización
└── storage/         — Serialización a .npz y .json
```

---

## API de alto nivel

```python
# Captura + métricas de un solo texto
run, metrics = aspec.capture_and_compute(adapter, input_ids, style_idx=0)

# Batch de textos
runs, metrics_list = aspec.capture_batch(adapter, ids_list, style_idx=0)

# Experimento de degradación completo (genera 6 variantes automáticamente)
per_cond = aspec.run_degradation_exp(
    adapter, base_ids, vocab_size=32000, prefix_len=3, style_idx=0
)

# Comparación de dos estilos sobre el mismo conjunto de textos
metrics_wiki, metrics_poem = aspec.run_style_comparison(
    adapter, ids_list, style_a=0, style_b=1
)
```

---

## Tests

```bash
pytest tests/ -v
# 97 tests: perturbaciones, métricas espectrales, normalización y serialización
```

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| `docs/concepts/metrics.md` | Definición matemática e interpretación de las 14 métricas |
| `docs/concepts/capture.md` | Arquitectura del pipeline de captura y shapes canónicos |
| `docs/concepts/viz.md` | Referencia completa de las 6 funciones de visualización |

---