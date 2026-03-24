"""
Test de integración completo del paquete attnspectra con el GPT del TFG.

Ejecutar desde la raíz:
    python integration_test.py

Requisitos:
    - pip install -e ".[dev]"
    - checkpoints/best.pt  (checkpoint entrenado)
    - tokenizer_mix_es/tokenizer.json  (tokenizer entrenado)

Salidas generadas:
    - test_attention_matrix.png
    - test_degradation.png
    - test_run.npz
    - test_metrics.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Añadimos experiments/ al path para poder importar gpt_model
sys.path.insert(0, str(Path(__file__).parent / "experiments"))

import torch
from tokenizers import Tokenizer

import attnspectra as aspec
from experiments.models.gpt_model import GPT


# Configuración

CKPT_PATH = "experiments/checkpoints/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# 1. Carga del checkpoint

print("\n[1/7] Cargando checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

TOKENIZER_PATH = Path(__file__).parent / "experiments" / "tokenizer_mix_es" / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# IDs especiales guardados en el checkpoint
sid = ckpt["special_ids"]
pad_id      = sid["pad_id"]
bos_id      = sid["bos_id"]
eos_id      = sid["eos_id"]
wiki_tok_id = sid["wiki_tok_id"]
poem_tok_id = sid["poem_tok_id"]
sep_id      = sid["sep_id"]

cfg = ckpt["config"]
model = GPT(**cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
torch.set_grad_enabled(False)

print(f"  Checkpoint cargado en opt_step={ckpt['opt_step']}")
print(f"  Val metrics: {ckpt.get('extra_metrics', {})}")


# 2. Adapter

print("\n[2/7] Creando adapter...")
adapter = aspec.CustomGPTAdapter(
    model=model,
    tokenizer=tokenizer,
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    d_model=cfg["d_model"],
    model_name="custom-gpt-es",
    device=device,
)
print(f"  Adapter OK: {adapter.model_info}")

# 3. Captura

print("\n[3/7] Capturando atención...")
config = aspec.CaptureConfig(
    capture_attn_pre=True,
    capture_scores=False,
    target_layers=None,
)

text = "La fotosíntesis convierte la luz solar en energía química."

# Construimos el prefijo igual que en el notebook: <bos> <wiki> <sep> + texto
prefix_ids = [bos_id, wiki_tok_id, sep_id]
content_ids = tokenizer.encode(text.replace("\n", " <nl> ")).ids
input_ids = torch.tensor([prefix_ids + content_ids], dtype=torch.long)

run = adapter.capture_from_ids(input_ids, config, style_idx=0)
print(f"  CapturedRun OK: {run.n_layers} capas, seq_len={run.seq_len}, "
      f"batch={run.batch_size}")


# 4. Métricas

print("\n[4/7] Calculando métricas espectrales...")
metrics = aspec.compute_head_metrics(run)
print(f"  HeadMetrics shape: {metrics.A_attn_entropy.shape}")  # (n_layers, n_heads)
print(f"  A_attn_entropy media (capa 3): {metrics.A_attn_entropy[3].mean():.4f}")
print(f"  A_effective_rank media (capa 3): {metrics.A_effective_rank[3].mean():.4f}")


# 5. Visualización

print("\n[5/7] Generando visualizaciones...")

# Matriz de atención individual
A = run.get_attention(layer=3, head=2)  # (Q, K)
fig = aspec.plot_attention_matrix(
    A, run.token_strs,
    max_tokens=50,
    title="Layer 3 · Head 2 | attn_pre"
)
fig.savefig("test_attention_matrix.png", dpi=100, bbox_inches="tight")
print("  Guardado: test_attention_matrix.png")

# Heatmap de métricas por capa × cabeza
fig2 = aspec.heatmap_metric(
    metrics.A_attn_entropy,
    title="A_attn_entropy por capa × cabeza",
    label="entropía"
)
fig2.savefig("test_heatmap_entropy.png", dpi=100, bbox_inches="tight")
print("  Guardado: test_heatmap_entropy.png")


# 6. IO round-trip

print("\n[6/7] Probando serialización...")
aspec.save_run(run, "test_run.npz")
run2 = aspec.load_run("test_run.npz")
assert run2.n_layers == run.n_layers, "n_layers no coincide tras round-trip"
assert run2.seq_len == run.seq_len,   "seq_len no coincide tras round-trip"
print(f"  save_run/load_run OK: {run2.n_layers} capas, seq_len={run2.seq_len}")

aspec.save_metrics(metrics, "test_metrics.json")
metrics2 = aspec.load_metrics("test_metrics.json")
assert metrics2.n_layers == metrics.n_layers
print(f"  save_metrics/load_metrics OK")


# 7. Experimento de perturbación

print("\n[7/7] Experimento de perturbación...")
aspec.set_seed(123)

base_ids = (prefix_ids + content_ids)
variants = aspec.make_variants(
    base_ids,
    vocab_size=cfg["vocab_size"],
    prefix_len=3,
    seed=123,
)

per_cond: dict[str, list] = {}
for name, vid in variants.items():
    vid_tensor = torch.tensor([vid], dtype=torch.long)
    vrun = adapter.capture_from_ids(vid_tensor, config, style_idx=0)
    per_cond[name] = [aspec.compute_head_metrics(vrun)]
    print(f"  variante '{name}' OK")

fig3 = aspec.plot_delta_lines_by_condition(
    per_cond,
    metric_key="A_attn_entropy",
    title="Δ entropía de atención vs capa (degradado − clean)"
)
fig3.savefig("test_degradation.png", dpi=100, bbox_inches="tight")
print("  Guardado: test_degradation.png")


# Resumen

print("\n" + "="*55)
print("✓ Integración completa OK")
print("="*55)
print("Ficheros generados:")
print("  test_attention_matrix.png")
print("  test_heatmap_entropy.png")
print("  test_run.npz")
print("  test_metrics.json")
print("  test_degradation.png")