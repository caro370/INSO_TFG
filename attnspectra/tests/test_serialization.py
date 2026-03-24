"""
Tests de serialización para storage/save.py y storage/load.py.

Comprueba que el ciclo save → load es un round-trip exacto para
CapturedRun y HeadMetrics. No requiere GPU ni modelo real.
"""

import json
from pathlib import Path

import numpy as np
import torch

from attnspectra.core.types import CapturedRun, HeadMetrics, ModelInfo
from attnspectra.storage.formats import SCHEMA_VERSION
from attnspectra.storage.load import load_metrics, load_run
from attnspectra.storage.save import save_metrics, save_run



def make_model_info() -> ModelInfo:
    return ModelInfo(
        name="test-gpt",
        architecture="decoder",
        n_layers=4,
        n_heads=2,
        d_model=64,
    )


def make_captured_run(
    n_layers: int = 4,
    n_heads: int = 2,
    seq_len: int = 10,
    batch_size: int = 1,
    sparse: bool = False,
) -> CapturedRun:
    """
    Crea un CapturedRun sintético.

    Parameters
    ----------
    sparse:
        Si True, algunas capas tendrán atención None (para simular
        target_layers parcial).
    """
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    attentions = []
    for i in range(n_layers):
        if sparse and i % 2 == 1:
            attentions.append(None)
        else:
            A = torch.rand(batch_size, n_heads, seq_len, seq_len)
            A = A / A.sum(dim=-1, keepdim=True)
            attentions.append(A)

    token_strs = [f"tok_{i}" for i in range(seq_len)]

    return CapturedRun(
        input_ids=input_ids,
        attentions=attentions,
        scores=None,
        token_strs=token_strs,
        model_info=make_model_info(),
        style_idx=1,
    )


def make_head_metrics(n_layers: int = 4, n_heads: int = 2, seq_len: int = 10) -> HeadMetrics:
    """Crea un HeadMetrics sintético con valores aleatorios."""
    rng = np.random.default_rng(42)
    shape = (n_layers, n_heads)

    return HeadMetrics(
        A_attn_entropy=rng.random(shape),
        A_effective_rank=rng.random(shape) * n_heads,
        A_top_singular=rng.random(shape),
        A_spectral_decay_rate=rng.random(shape),
        A_anisotropy_index=rng.random(shape),
        A_gini=rng.random(shape),
        A_attn_distance=rng.random(shape),
        S_effective_rank=rng.random(shape) * n_heads,
        S_top_singular=rng.random(shape),
        Sc_effective_rank=rng.random(shape) * n_heads,
        Sc_top_singular=rng.random(shape),
        Sc_spectral_decay_rate=rng.random(shape),
        Sc_anisotropy_index=rng.random(shape),
        Sc_gini=rng.random(shape),
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
    )

class TestSaveLoadRun:
    def test_round_trip_input_ids(self, tmp_path):
        run = make_captured_run()
        path = tmp_path / "run.npz"
        save_run(run, path)
        run2 = load_run(path)
        assert torch.equal(run.input_ids, run2.input_ids)

    def test_round_trip_attentions(self, tmp_path):
        run = make_captured_run()
        path = tmp_path / "run.npz"
        save_run(run, path)
        run2 = load_run(path)
        for i, (A, A2) in enumerate(zip(run.attentions, run2.attentions)):
            if A is None:
                assert A2 is None
            else:
                assert torch.allclose(A, A2, atol=1e-6), f"Capa {i} no coincide"

    def test_round_trip_n_layers(self, tmp_path):
        run = make_captured_run(n_layers=6)
        path = tmp_path / "run.npz"
        save_run(run, path)
        run2 = load_run(path)
        assert run2.n_layers == run.n_layers

    def test_round_trip_seq_len(self, tmp_path):
        run = make_captured_run(seq_len=15)
        path = tmp_path / "run.npz"
        save_run(run, path)
        run2 = load_run(path)
        assert run2.seq_len == run.seq_len

    def test_round_trip_token_strs(self, tmp_path):
        run = make_captured_run()
        path = tmp_path / "run.npz"
        save_run(run, path)
        run2 = load_run(path)
        assert run2.token_strs == run.token_strs

    def test_round_trip_style_idx(self, tmp_path):
        run = make_captured_run()
        path = tmp_path / "run.npz"
        save_run(run, path)
        run2 = load_run(path)
        assert run2.style_idx == run.style_idx

    def test_round_trip_model_info(self, tmp_path):
        run = make_captured_run()
        path = tmp_path / "run.npz"
        save_run(run, path)
        run2 = load_run(path)
        assert run2.model_info.name == run.model_info.name
        assert run2.model_info.n_layers == run.model_info.n_layers
        assert run2.model_info.n_heads == run.model_info.n_heads

    def test_sparse_attentions_preserved(self, tmp_path):
        """Las capas con atención None deben seguir siendo None tras round-trip."""
        run = make_captured_run(n_layers=4, sparse=True)
        path = tmp_path / "run_sparse.npz"
        save_run(run, path)
        run2 = load_run(path)
        for i in range(run.n_layers):
            if run.attentions[i] is None:
                assert run2.attentions[i] is None, f"Capa {i} debería ser None"
            else:
                assert run2.attentions[i] is not None, f"Capa {i} no debería ser None"

    def test_file_is_npz(self, tmp_path):
        run = make_captured_run()
        path = tmp_path / "run.npz"
        save_run(run, path)
        assert path.exists()
        # Un .npz es un ZIP; verificamos que el fichero empieza con la firma ZIP
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic[:2] == b"PK"

    def test_schema_version_in_file(self, tmp_path):
        """La versión del esquema debe estar guardada en el metadata."""
        run = make_captured_run()
        path = tmp_path / "run.npz"
        save_run(run, path)
        data = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(data["_meta"]).decode("utf-8"))
        assert meta["schema_version"] == SCHEMA_VERSION

    def test_path_as_string(self, tmp_path):
        """save_run y load_run deben aceptar strings además de Path."""
        run = make_captured_run()
        path_str = str(tmp_path / "run.npz")
        save_run(run, path_str)
        run2 = load_run(path_str)
        assert run2.n_layers == run.n_layers

    def test_batch_size_preserved(self, tmp_path):
        run = make_captured_run(batch_size=2, seq_len=8)
        path = tmp_path / "run_batch.npz"
        save_run(run, path)
        run2 = load_run(path)
        assert run2.batch_size == run.batch_size


class TestSaveLoadMetrics:
    def test_round_trip_arrays(self, tmp_path):
        metrics = make_head_metrics()
        path = tmp_path / "metrics.json"
        save_metrics(metrics, path)
        metrics2 = load_metrics(path)
        for key in metrics.as_dict():
            orig = getattr(metrics,  key)
            back = getattr(metrics2, key)
            assert np.allclose(orig, back, atol=1e-6), f"Campo {key} no coincide"

    def test_round_trip_scalars(self, tmp_path):
        metrics = make_head_metrics(n_layers=3, n_heads=4, seq_len=20)
        path = tmp_path / "metrics.json"
        save_metrics(metrics, path)
        metrics2 = load_metrics(path)
        assert metrics2.n_layers == metrics.n_layers
        assert metrics2.n_heads  == metrics.n_heads
        assert metrics2.seq_len  == metrics.seq_len

    def test_output_is_valid_json(self, tmp_path):
        metrics = make_head_metrics()
        path = tmp_path / "metrics.json"
        save_metrics(metrics, path)
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_schema_version_present(self, tmp_path):
        metrics = make_head_metrics()
        path = tmp_path / "metrics.json"
        save_metrics(metrics, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == SCHEMA_VERSION

    def test_all_metric_keys_present(self, tmp_path):
        metrics = make_head_metrics()
        path = tmp_path / "metrics.json"
        save_metrics(metrics, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        expected_keys = {
            "A_attn_entropy", "A_effective_rank",
            "A_top_singular", "S_effective_rank",
            "S_top_singular",
        }
        assert expected_keys.issubset(data.keys())

    def test_path_as_string(self, tmp_path):
        metrics = make_head_metrics()
        path_str = str(tmp_path / "metrics.json")
        save_metrics(metrics, path_str)
        metrics2 = load_metrics(path_str)
        assert metrics2.n_layers == metrics.n_layers

    def test_nan_values_survive_round_trip(self, tmp_path):
        """Las capas no capturadas (NaN) deben sobrevivir el round-trip."""
        metrics = make_head_metrics(n_layers=4)
        metrics.A_attn_entropy[1, :] = np.nan   # simula capa no capturada
        path = tmp_path / "metrics_nan.json"
        save_metrics(metrics, path)
        metrics2 = load_metrics(path)
        assert np.isnan(metrics2.A_attn_entropy[1, :]).all()