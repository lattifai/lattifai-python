#!/usr/bin/env python3
"""Export torchaudio Resample transforms as ONNX models.

Usage:
    python scripts/export_resampler.py

Generates ONNX files for common sample rates → 16kHz in:
    src/lattifai/data/resamplers/resampler_{sr}.onnx
"""

from pathlib import Path

import onnx
import torch
import torchaudio

SAMPLE_RATES = [8000, 22050, 24000, 32000, 44100, 48000]
TARGET_SR = 16000
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "lattifai" / "data" / "resamplers"


class ResamplerWrapper(torch.nn.Module):
    """Wrap torchaudio.transforms.Resample for ONNX export."""

    def __init__(self, source_sr: int, target_sr: int):
        super().__init__()
        self.resample = torchaudio.transforms.Resample(source_sr, target_sr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


def export_resampler(source_sr: int, target_sr: int, output_path: Path) -> None:
    model = ResamplerWrapper(source_sr, target_sr)
    model.eval()

    # Use 1 second of audio as dummy input (single channel)
    dummy = torch.randn(1, source_sr)

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {1: "num_samples"},
            "output": {1: "num_samples"},
        },
        opset_version=17,
    )

    # Inline external data into the .onnx file and remove .data files
    onnx_model = onnx.load(str(output_path), load_external_data=True)

    # Fix output shape: torch.onnx.export records a fixed dim_value from the
    # dummy input even with dynamic_axes, because torchaudio's Resample uses
    # shape-dependent ops.  Clear the fixed value and set a symbolic dim_param
    # so onnxruntime won't emit VerifyOutputSizes warnings at inference time.
    for output in onnx_model.graph.output:
        shape = output.type.tensor_type.shape
        if shape and len(shape.dim) > 1:
            dim = shape.dim[1]
            dim.Clear()
            dim.dim_param = "num_samples"

    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)
    data_path = Path(str(output_path) + ".data")
    if data_path.exists():
        data_path.unlink()

    # Verify exported model
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": dummy.numpy()})[0]
    print(f"  Verified: input({source_sr}, {dummy.shape[1]}) -> output({target_sr}, {result.shape[1]})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting resamplers to {OUTPUT_DIR}")

    for sr in SAMPLE_RATES:
        output_path = OUTPUT_DIR / f"resampler_{sr}.onnx"
        print(f"Exporting {sr}Hz -> {TARGET_SR}Hz ...")
        export_resampler(sr, TARGET_SR, output_path)
        size_kb = output_path.stat().st_size / 1024
        print(f"  -> {output_path.name} ({size_kb:.1f} KB)")

    print("\nDone! All resamplers exported.")


if __name__ == "__main__":
    main()
