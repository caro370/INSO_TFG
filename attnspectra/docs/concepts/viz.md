# viz/ — Referencia de visualizaciones

Todas las funciones del módulo `viz/` devuelven un objeto `matplotlib.Figure`
y aceptan un parámetro opcional `ax: plt.Axes` para incrustarlas en figuras
compuestas. Si `ax` se omite, cada función crea su propia figura.

---

## Índice

1. [Funciones](#1-funciones)
   - [`plot_attention_matrix`](#plot_attention_matrix)
   - [`plot_attention_matrix_interactive`](#plot_attention_matrix_interactive)
   - [`heatmap_metric`](#heatmap_metric)
   - [`heatmap_delta`](#heatmap_delta)
   - [`plot_metric_by_layer`](#plot_metric_by_layer)
   - [`plot_delta_lines_by_condition`](#plot_delta_lines_by_condition)
2. [Cuándo usar cada visualización](#3-cuándo-usar-cada-visualización)

---

## 1. Funciones

### `plot_attention_matrix`

```python
from attnspectra.viz import plot_attention_matrix

fig = plot_attention_matrix(
    A,            # Tensor o array (T, T) — matriz de atención de una cabeza
    token_strs,   # list[str] — etiquetas de los tokens
    title="",
    max_tokens=40,
    cmap="Blues",
    ax=None,
)
```

Dibuja la matriz de atención de una sola cabeza como heatmap. Los ejes
muestran los tokens de entrada (eje X, keys) y los tokens objetivo (eje Y,
queries). Cada celda `(i, j)` indica cuánto presta atención el token `i`
al token `j`.

El parámetro `max_tokens` trunca la secuencia a las primeras `max_tokens`
posiciones para mantener el gráfico legible en textos largos. Las etiquetas
se escapan y truncan internamente con `format_token_labels`.

**Cuándo usarla:** análisis cualitativo de una cabeza concreta. Útil para
detectar patrones como atención diagonal (tokens cercanos), atención al
primer token ([CLS] o BOS) o patrones sintácticos específicos.

---

### `plot_attention_matrix_interactive`

```python
from attnspectra.viz import plot_attention_matrix_interactive

fig = plot_attention_matrix_interactive(
    run,
    max_tokens=50,
    title="",
    colorscale="Blues",
    width= 750,
    height = 650,
)
```

Es la versión interactiva de la visualización anterior. Con los sliders se puede 
elegir la capa y la cabeza de la que se quiere ver el mapa de atención del modelo. 
La principal diferencia entre ambas es que `plot_attention_matrix` se utilizar para 
analizar en profundidad una cabeza concreta y `plot_attention_matrix_interactive`se 
usa para explorar las cabezas. 

**Cuándo usarla:** para explorar la atención del modelo en todas las capas y cabezas.
Puedes recorrer todas las combinaciones (capa, cabeza) con los sliders y detectar 
visualmente patrones llamativos antes de hacer un análisis cuantitativo con las métricas.
También es útil para verificar que la captura de atención es correcta. 
---

### `heatmap_metric`

```python
from attnspectra.viz import heatmap_metric

fig = heatmap_metric(
    M,            # array (n_layers, n_heads)
    title="",
    cmap="plasma",
    xlabel="Cabeza",
    ylabel="Capa",
    ax=None,
)
```

Heatmap de una métrica escalar en el espacio (capa × cabeza). Cada celda
contiene el valor numérico de la métrica para esa combinación (capa, cabeza).
El colormap secuencial `plasma` es el defecto porque las métricas de
`HeadMetrics` son no negativas en general.

**Cuándo usarla:** visión general de cómo varía una sola métrica en todo
el modelo. Permite identificar capas o cabezas con valores extremos.

---

### `heatmap_delta`

```python
from attnspectra.viz import heatmap_delta

fig = heatmap_delta(
    A,            # array (n_layers, n_heads) — condición A
    B,            # array (n_layers, n_heads) — condición B
    title="",
    cmap="RdBu_r",
    ax=None,
)
```

Heatmap de la diferencia `A − B` en el espacio (capa × cabeza). El colormap
divergente `RdBu_r` centra el cero en blanco: rojo indica que A es mayor
y azul indica que B es mayor.

**Cuándo usarla:** comparación directa de dos condiciones (por ejemplo,
texto coherente vs. texto ruido, estilo wiki vs. estilo poem). Permite
identificar qué cabezas y capas reaccionan más al cambio de condición.

---

### `plot_metric_by_layer`

```python
from attnspectra.viz import plot_metric_by_layer

fig = plot_metric_by_layer(
    metrics_list,    # list[HeadMetrics]
    metric_key,      # str — campo de HeadMetrics
    title="",
    label="",
    color=None,
    ax=None,
)
```

Traza la media de una métrica sobre las cabezas vs. capa, con una banda
sombreada de ±1 desviación estándar. Si se pasa una lista con varios
`HeadMetrics`, cada uno genera su propia línea.

**Cuándo usarla:** estudiar la evolución de una métrica a lo largo de las
capas del modelo. Revela gradientes como el aumento progresivo de
`A_attn_distance` (las capas altas tienden a atender más lejos) o la
caída de `A_attn_entropy` en capas especializadas.

---

### `plot_delta_lines_by_condition`

```python
from attnspectra.viz import plot_delta_lines_by_condition

fig = plot_delta_lines_by_condition(
    per_condition,   # dict[str, list[HeadMetrics]] — una entrada por condición
    metric_key,      # str
    base,            # str — nombre de la condición base (el Δ se calcula respecto a ella)
    title="",
    ax=None,
)
```

Para cada condición distinta de `base`, calcula `Δ = media(condición) − media(base)`
promediado sobre textos y cabezas, y lo traza vs. capa. Produce una línea
por condición, lo que permite ver cuánto y en qué capas cada tipo de
degradación afecta a la métrica.

**Cuándo usarla:** experimentos de degradación de señal (`make_variants`).
Responde preguntas como: ¿en qué capas afecta más el barajar los tokens
frente a reemplazarlos aleatoriamente?

---

## 2. Cuándo usar cada visualización

Esta tabla orienta la elección de función según la pregunta de análisis:

| Pregunta | Función recomendada |
|----------|---------------------|
| ¿Cómo varía **una métrica** en todo el modelo (capa × cabeza)? | `heatmap_metric` |
| ¿Qué cabezas cambian más entre dos condiciones globalmente? | `heatmap_delta` |
| ¿Cómo evoluciona una métrica a lo largo de las capas? | `plot_metric_by_layer` |
| ¿Cómo responde cada capa a los distintos tipos de degradación? | `plot_delta_lines_by_condition` |
| ¿Qué matriz de atención tiene una cabeza concreta? | `plot_attention_matrix` |
| ¿Qué matriz de atención tiene cada cabeza en cada capa? | `plot_attention_matrix_interactive` |

### Notas sobre tipos de entrada

Las funciones se dividen en dos grupos según el tipo de dato que necesitan:

**Aceptan `HeadMetrics`** (solo los valores escalares calculados):
`heatmap_metric`, `heatmap_delta`, `plot_metric_by_layer`,
`plot_delta_lines_by_condition`.

**Requieren `CapturedRun`** (necesitan los tensores de atención originales):
`plot_attention_matrix`, `plot_attention_matrix_interactive`.
