# attnspectra

**attnspectra** es una librería para analizar el comportamiento de la atención en modelos transformer mediante métricas espectrales, entropía y visualizaciones.

---

## Instalación

```bash
pip install attnspectra
```

Extras opcionales:

```bash
# Soporte para modelos Hugging Face
pip install attnspectra[hf]

# Visualización interactiva (Plotly + widgets)
pip install attnspectra[viz]

# Todo incluido
pip install attnspectra[all]

# Dependencias de desarrollo
pip install attnspectra[dev]
---

## Uso rápido

```python
import attnspectra as aspec

# Suponiendo que ya tienes un modelo y tokenizer
model = ...
tokenizer = ...

# Crear adapter
adapter = aspec.CustomGPTAdapter(
    model=model,
    tokenizer=tokenizer,
    n_layers=8,
    n_heads=8,
    d_model=512,
)

# Capturar atención y calcular métricas
run, metrics = aspec.capture_and_compute(adapter, input_ids)

# Acceder a métricas (shape: n_layers × n_heads)
print(metrics.A_attn_entropy)
print(metrics.A_effective_rank)
print(metrics.A_attn_distance)

# Visualizar atención
A, tokens = aspec.get_content_attention(run, layer=0, head=0)
fig = aspec.plot_attention_matrix(A, tokens)
```

```md
> Nota: las visualizaciones interactivas requieren dependencias opcionales.
> Instálalas con:
> `pip install attnspectra[viz]`
```
---

## Qué puedes hacer con attnspectra

* Analizar patrones de atención en modelos transformer
* Estudiar estructura espectral de matrices de atención
* Comparar comportamiento entre diferentes inputs o estilos
* Visualizar matrices de atención y métricas por capa y cabeza

---

## Funcionalidades principales

* **Captura de atención** mediante adapters flexibles
* **14 métricas** (entropía, rango efectivo, espectro, anisotropía, etc.)
* **Análisis por capa y cabeza**
* **Visualizaciones** con matplotlib y modo interactivo con Plotly (opcional)
* **Experimentos automatizados** (degradación, comparación de estilos)

---

## Documentación completa

Consulta el repositorio para ejemplos, teoría y documentación detallada:

https://github.com/caro370/INSO_TFG

---

## Requisitos

* Python ≥ 3.10
* PyTorch ≥ 2.1
* NumPy ≥ 1.24
* Matplotlib ≥ 3.7
