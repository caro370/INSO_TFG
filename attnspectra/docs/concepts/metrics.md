# Métricas espectrales de atención

Este documento explica las métricas que calcula `attnspectra.compute_head_metrics()` y su interpretación en el contexto del análisis de modelos transformer.

---

## Notación

Sea **A** ∈ ℝ^(Q×K) la matriz de atención de una cabeza concreta, donde:
- **Q** = número de posiciones query (tokens de entrada)
- **K** = número de posiciones key (igual a Q en atención causal)
- Cada fila de **A** es una distribución de probabilidad: Σⱼ Aᵢⱼ = 1
- **σ₁ ≥ σ₂ ≥ ... ≥ σᵣ** son los valores singulares de **A**, con r = min(Q, K)
- **pᵢ = σᵢ / Σⱼ σⱼ** es la distribución normalizada sobre valores singulares

---

## Las tres familias de métricas

`compute_head_metrics()` calcula 14 métricas organizadas en tres familias según la matriz sobre la que opera.

### Familia A — sobre la matriz de atención post-softmax

**A** ∈ [0, 1] con filas que suman 1. Es la salida directa del softmax. Mide la estructura de la distribución de probabilidad de atención. Siempre disponible.

### Familia S — sobre S = log(A + ε)

Al aplicar logaritmo, los valores pasan de [0, 1] a (-∞, 0]. Amplifica diferencias en probabilidades bajas que en **A** son casi invisibles. Por ejemplo: `A = [0.001, 0.002, 0.997]` parece dominado por el tercer token, pero `S = [-6.9, -6.2, -0.003]` revela estructura en los dos primeros. Aproxima la geometría del espacio de scores sin capturar los logits explícitamente. Siempre disponible.

### Familia Sc — sobre los scores Q·K^T reales (pre-softmax)

Son los logits antes del softmax. Teóricamente la opción más correcta para análisis espectral porque no está distorsionada por la normalización. Solo disponible si se capturó con `capture_scores=True` en `CaptureConfig`. Los campos Sc_* contienen NaN si los scores no fueron capturados. Usar `metrics.has_scores_metrics()` para comprobarlo.

---

## Familia A — métricas sobre la atención post-softmax

### `A_attn_entropy` — Entropía de atención

**Definición:**

$$H(A) = -\frac{1}{Q} \sum_{i=1}^{Q} \sum_{j=1}^{K} A_{ij} \log A_{ij}$$

**Interpretación:**
- **Alto** → atención difusa sobre muchas posiciones (cabeza generalista o de integración global)
- **Bajo** → atención concentrada en pocas posiciones (cabeza especializada o sintáctica)
- **Rango teórico:** [0, log(K)]

---

### `A_effective_rank` — Rango efectivo (continuo, Roy & Vetterli 2007)

**Definición:**

$$\text{erank}(A) = \exp\!\left(-\sum_i p_i \log p_i\right) = \exp(H(\mathbf{p}))$$

**Interpretación:**
- **Alto** (cercano a min(Q,K)) → valores singulares uniformes, la cabeza captura muchos patrones ortogonales independientes
- **Bajo** (cercano a 1) → un solo valor singular domina, la matriz es casi de rango 1
- **Rango teórico:** [1, min(Q, K)]

---

### `A_top_singular` — Valor singular dominante (radio espectral)

**Definición:** 
$${\sigma_1 = ‖A‖_2}$$

**Interpretación:**
- **Alto** → una dirección concentra la mayor parte de la energía de la transformación
- **Bajo** → energía distribuida entre muchas direcciones

---

### `A_spectral_decay_rate` — Velocidad de caída espectral (SDR)

**Definición:** pendiente α del ajuste log-lineal por mínimos cuadrados:

$$\log(\sigma_i) = -\alpha \cdot i + \beta$$

**Interpretación:**
- **α alto** → caída rápida, compresión agresiva, pocas direcciones dominantes
- **α bajo** → caída suave, uso más uniforme de todas las direcciones de transformación
- Unidades: nats por índice de valor singular
---

### `A_anisotropy_index` — Índice de anisotropía

**Definición:**

$$AI = \frac{\sigma_{\max} - \sigma_{\min}}{\sigma_{\text{mean}}}$$

**Interpretación:**
- **Alto** → ciertas direcciones de entrada se amplifican mucho más que otras (procesamiento anisotrópico, sensibilidades especializadas)
- **Bajo** (≈ 0) → procesamiento isotrópico, todas las direcciones tratadas por igual
---

### `A_gini` — Coeficiente de Gini espectral

**Definición** (con σᵢ ordenados de menor a mayor, i = 1..K):

$$G = \frac{2 \sum_i i \cdot \sigma_i}{K \sum_i \sigma_i} - \frac{K+1}{K}$$

**Interpretación:**
- **G = 0** → distribución perfectamente igualitaria (todos los sv iguales)
- **G → 1** → concentración extrema (un sv domina completamente)

Más sensible que la entropía a concentraciones extremas. Una cabeza con Gini alto es altamente especializada. Una con Gini bajo tiene procesamiento más generalista.
---

### `A_attn_distance` — Distancia media de atención

**Definición** (Kovaleva et al., 2019):

$$\bar{D} = \frac{\sum_{i=1}^{T} \sum_{j=1}^{i} \alpha_{ij} \cdot (i-j)}{\sum_{i=1}^{T} \sum_{j=1}^{i} \alpha_{ij}}$$

En modo causal (por defecto) solo se considera la parte triangular inferior (j ≤ i). Para modelos bidireccionales usar `causal=False`, que emplea |i-j| sobre toda la matriz.

**Interpretación:**
- **Bajo** → la cabeza atiende a tokens cercanos (atención local, típico en capas bajas)
- **Alto** → la cabeza atiende a tokens lejanos (atención a larga distancia, típico en capas profundas)
- **Unidades:** número de tokens

---

## Familia S — métricas sobre log(A)

### `S_effective_rank`, `S_top_singular`

Misma definición que sus equivalentes de la familia A, pero calculados sobre **S = log(A + ε)**.

**Cuándo usar S en lugar de A:**
- Las métricas A son más intuitivas y directamente interpretables
- Las métricas S son más sensibles a cambios en la distribución de probabilidades bajas
- Comparar A y S puede revelar cuánto comprime el softmax la estructura espectral real

---

## Familia Sc — métricas sobre scores Q·K^T

### `Sc_effective_rank`, `Sc_top_singular`, `Sc_spectral_decay_rate`, `Sc_anisotropy_index`, `Sc_gini`

Misma definición que sus equivalentes de la familia A, pero calculados sobre los logits Q·K^T antes del softmax.

**Activar la familia Sc:**

```python
from attnspectra import CaptureConfig, compute_head_metrics

config = CaptureConfig(capture_attn_pre=True, capture_scores=True)
run = adapter.capture_from_ids(input_ids, config)
metrics = compute_head_metrics(run)

if metrics.has_scores_metrics():
    print(metrics.Sc_effective_rank)  # (n_layers, n_heads)
```

**Por qué son útiles:**

La comparación directa entre la familia A y la familia Sc permite cuantificar la distorsión que introduce el softmax:

```python
# Diferencia de rango efectivo: efecto del softmax
delta_er = metrics.A_effective_rank - metrics.Sc_effective_rank
```

---

## Tabla resumen

| Campo | Familia | Rango teórico | Alto significa | Bajo significa |
|-------|---------|---------------|----------------|----------------|
| `A_attn_entropy` | A | [0, log(K)] | Atención difusa | Atención focalizada |
| `A_effective_rank` | A | [1, min(Q,K)] | Muchos patrones | Un patrón dominante |
| `A_top_singular` | A | [0, ∞) | Energía concentrada | Energía distribuida |
| `A_spectral_decay_rate` | A | [0, ∞) | Compresión agresiva | Uso uniforme de dirs. |
| `A_anisotropy_index` | A | [0, ∞) | Sesgo direccional | Procesamiento isotrópico |
| `A_gini` | A | [0, 1) | Alta especialización | Distribución igualitaria |
| `A_attn_distance` | A | [0, T-1] | Atención lejana | Atención local |
| `S_effective_rank` | S | [1, min(Q,K)] | Scores distribuidos | Score dominante |
| `S_top_singular` | S | [0, ∞) | Score dominante | Scores similares |
| `Sc_*` | Sc | igual que A | igual que A | igual que A |
