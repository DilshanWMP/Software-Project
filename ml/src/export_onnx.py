"""Export findings / gate models to ONNX + json metadata (internal/AI_CONTRACT.md).

Dev (untrained) models for the C#/UI work:
    ml\\.venv\\Scripts\\python.exe ml\\src\\export_onnx.py --dev-untrained

Real findings model (after train.py + evaluate.py):
    ml\\.venv\\Scripts\\python.exe ml\\src\\export_onnx.py --task findings ^
        --checkpoint ml\\checkpoints\\<run>\\best.pt --report ml\\reports\\<run>\\report.json

Real gate model:
    ml\\.venv\\Scripts\\python.exe ml\\src\\export_onnx.py --task gate ^
        --checkpoint ml\\checkpoints\\<gate_run>\\best.pt --report ml\\reports\\<gate_run>\\report.json

Every export checks onnxruntime vs torch parity (max abs diff) and the file size.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AIMODELS_DIR, DEFAULT_QUALITY, PREPROCESS_BLOCK, REPORTS_DIR, SUPER_CLASSES,  # noqa: E402
                    load_json, save_json, seed_everything)
from models import FindingsExport, GateExport, build_model, check_cam_exact  # noqa: E402

OPSET = 17
MAX_MB = 50.0
PARITY_TOL = 1e-4


def _now() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def export_graph(wrapper: torch.nn.Module, size: int, out_path: Path, output_names: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    dummy = torch.rand(1, 3, size, size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(input_names=["input"], output_names=output_names,
                      opset_version=OPSET, do_constant_folding=True)
        try:
            torch.onnx.export(wrapper, (dummy,), str(out_path), dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo kwarg
            torch.onnx.export(wrapper, (dummy,), str(out_path), **kwargs)
    # Make sure everything is in one file (no external data).
    import onnx
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    ext = [t.name for t in m.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    if ext:
        raise RuntimeError(f"ONNX uses external data for {len(ext)} tensors")


def verify(wrapper: torch.nn.Module, onnx_path: Path, size: int, n: int = 3) -> dict:
    """Compare onnxruntime CPU vs torch on n random + 1 structured input."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    assert inp.name == "input", inp.name
    assert list(inp.shape) == [1, 3, size, size], inp.shape
    names = [o.name for o in sess.get_outputs()]
    g = torch.Generator().manual_seed(0)
    xs = [torch.rand(1, 3, size, size, generator=g) for _ in range(n)]
    yy = torch.linspace(0, 1, size).view(1, 1, size, 1).expand(1, 3, size, size).contiguous()
    xs.append(yy)
    worst = {k: 0.0 for k in names}
    scale = {k: 0.0 for k in names}
    wrapper.eval()
    with torch.no_grad():
        for x in xs:
            t_out = wrapper(x)
            if isinstance(t_out, torch.Tensor):
                t_out = (t_out,)
            o_out = sess.run(None, {"input": x.numpy()})
            for k, t, o in zip(names, t_out, o_out):
                assert tuple(t.shape) == o.shape, (k, t.shape, o.shape)
                worst[k] = max(worst[k], float(np.abs(t.numpy() - o).max()))
                scale[k] = max(scale[k], float(np.abs(t.numpy()).max()))
    size_mb = onnx_path.stat().st_size / 1e6
    res = {"outputs": names, "shapes": [list(o.shape) for o in sess.get_outputs()],
           "max_abs_diff": worst, "max_abs_value": scale, "size_mb": round(size_mb, 2)}
    print(f"[verify] {onnx_path.name}: {res}")
    if size_mb > MAX_MB:
        raise RuntimeError(f"{onnx_path.name} is {size_mb:.1f} MB > {MAX_MB} MB")
    # logits: absolute tolerance 1e-4. cam: values reach ~20 and only drive the heatmap
    # picture, so it is checked relative to its max |value| (1e-4).
    for k, v in worst.items():
        if k == "cam":
            rel = v / max(scale[k], 1e-6)
            res["cam_rel_diff"] = rel
            if rel > 1e-4:
                raise RuntimeError(f"Parity failed for cam: abs {v:.3g}, rel {rel:.3g}")
        elif v > PARITY_TOL:
            raise RuntimeError(f"Parity failed for {k}: abs {v:.3g} > {PARITY_TOL}")
    return res


def findings_json(model_version, arch, size, keys, temperature=1.0, uncertain_below=0.55,
                  thresholds=None, quality=None, trained_on="") -> dict:
    thresholds = thresholds or {}
    classes = []
    for i, k in enumerate(keys):
        info = SUPER_CLASSES[k]
        classes.append({"index": i, "key": k, "display": info["display"],
                        "abnormal": info["abnormal"], "threshold": float(thresholds.get(k, 0.5))})
    return {
        "model_version": model_version,
        "architecture": arch,
        "input_size": int(size),
        "temperature": float(temperature),
        "uncertain_below": float(uncertain_below),
        "classes": classes,
        "preprocess": dict(PREPROCESS_BLOCK),
        "quality": dict(quality or DEFAULT_QUALITY),
        "trained_on": trained_on,
        "created": _now(),
    }


def gate_json(model_version, arch, size, temperature=1.0, threshold=0.5) -> dict:
    return {"model_version": model_version, "architecture": arch, "input_size": int(size),
            "temperature": float(temperature), "endoscopy_threshold": float(threshold),
            "created": _now()}


def export_findings(model, arch, size, keys, out_dir: Path, meta: dict) -> dict:
    model.eval()
    cam_diff = check_cam_exact(model, size=min(size, 256))
    print(f"[findings] CAM exactness check: max |logits - (mean(cam)+b)| = {cam_diff:.2e}")
    wrap = FindingsExport(model).eval()
    onnx_path = out_dir / "gi_findings.onnx"
    export_graph(wrap, size, onnx_path, ["logits", "cam"])
    res = verify(wrap, onnx_path, size)
    save_json(meta, out_dir / "gi_findings.json")
    res["cam_exact_diff"] = cam_diff
    return res


def export_gate(model, size, out_dir: Path, meta: dict) -> dict:
    wrap = GateExport(model).eval()
    onnx_path = out_dir / "endo_gate.onnx"
    export_graph(wrap, size, onnx_path, ["logits"])
    res = verify(wrap, onnx_path, size)
    save_json(meta, out_dir / "endo_gate.json")
    return res


def load_checkpoint(path: Path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-untrained", action="store_true",
                    help="write untrained gi_findings_dev / endo_gate_dev models")
    ap.add_argument("--task", choices=["findings", "gate"], default="findings")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--report", type=Path, help="report.json from evaluate.py (temperature etc.)")
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--out-dir", type=Path, default=AIMODELS_DIR)
    args = ap.parse_args()
    seed_everything()

    if args.dev_untrained:
        keys = list(SUPER_CLASSES)
        m = build_model("efficientnet_b0", len(keys), pretrained=True)
        meta = findings_json("gi_findings_dev", "efficientnet_b0", 384, keys,
                             trained_on="UNTRAINED dev model (ImageNet backbone, random head)")
        r1 = export_findings(m, "efficientnet_b0", 384, keys, args.out_dir, meta)
        g = build_model("mobilenetv3_small_100", 2, pretrained=True)
        # Dev gate: shrink the random head and bias it toward "endoscopy" (p ~0.8-0.95) so
        # the app flow is not blocked by a random NotEndoscopy verdict during UI work.
        with torch.no_grad():
            fc = g.get_classifier()
            fc.weight.mul_(0.05)
            fc.bias.copy_(torch.tensor([0.0, 2.0]))
        r2 = export_gate(g, 224, args.out_dir, gate_json("endo_gate_dev", "mobilenetv3_small_100", 224))
        save_json({"gi_findings": r1, "endo_gate": r2}, REPORTS_DIR / "dev" / "export_verify.json")
        print("Dev models written to", args.out_dir)
        return

    if not args.checkpoint:
        ap.error("--checkpoint is required unless --dev-untrained")
    model, ck = load_checkpoint(args.checkpoint)
    rep = load_json(args.report) if args.report else {}
    cal = rep.get("calibration", {})
    if args.task == "findings":
        if ck.get("task", "findings") != "findings":
            raise SystemExit("checkpoint is not a findings model")
        mv = args.model_version or "gi_findings_v1"
        meta = findings_json(mv, ck["arch"], ck["img_size"], ck["classes"],
                             temperature=cal.get("temperature", 1.0),
                             uncertain_below=cal.get("uncertain_below", 0.55),
                             thresholds=cal.get("thresholds", {}),
                             quality=rep.get("quality_suggested", DEFAULT_QUALITY),
                             trained_on=ck.get("trained_on", "HyperKvasir labelled + GastroVision (super-classes)"))
        if not args.report:
            print("WARNING: no --report given; temperature=1, thresholds=0.5, default quality")
        res = export_findings(model, ck["arch"], ck["img_size"], ck["classes"], args.out_dir, meta)
    else:
        if ck.get("task") != "gate":
            raise SystemExit("checkpoint is not a gate model")
        mv = args.model_version or "endo_gate_v1"
        meta = gate_json(mv, ck["arch"], ck["img_size"],
                         temperature=cal.get("temperature", 1.0),
                         threshold=cal.get("endoscopy_threshold", 0.5))
        res = export_gate(model, ck["img_size"], args.out_dir, meta)
    print("Exported", mv, "to", args.out_dir, res)


if __name__ == "__main__":
    main()
