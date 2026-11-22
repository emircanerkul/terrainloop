# TerrainLoop — Executive Summary

**TerrainLoop** is an architecture and quantization-aware training proposal for compacting a 27B-depth dense transformer into a **9B looped core** using a mixed-precision `4-bit + 1-bit + 1-bit` (or `4-bit + ternary + ternary`) formulation.

## The problem

Pure 1-bit (`{-1, +1}`) and ternary (`{-1, 0, +1}`, 1.58-bit) models force every weight onto a **flat quantized surface**. All non-zero steps have identical magnitude, which crushes outlier channels, loses relative sensitivity, and typically requires training from scratch over trillions of tokens. Meanwhile, storing 27B distinct parameters at INT4 still demands 13.5 GB for weights plus a 27B-layer KV cache.

## The approach

TerrainLoop treats **loop/layer depth $i$ as a 4th dimension** (`looped = stacked`) and separates the role of precision across depth:

1. **Loop 0 ($i=0$) — 4-bit Bumped Terrain ($H_0$):** Stores a 9B backbone in 4-bit (16 elevation levels, 4.50 GB). It preserves outliers and fine relative magnitudes and can be warm-started from an existing pretrained 9B model.
2. **Loops 1 & 2 ($i=1, 2$) — Terrain-Coupled 1-Bit / Ternary ($T_1, T_2$):** Instead of adding a flat constant, each subsequent pass applies a deterministic 4D channel rotation/shift $\mathcal{D}_i(H_0)$ plus a mask $T_i \in \{-1, 0, +1\}$ multiplied by the local hill height $H_0$ and terrain slope $\nabla_i H_0$.
3. **Energy & KV Control:** Per-loop RMSNorm stabilizes variance across iterations, and cross-loop KV sharing keeps the inference KV cache at the 9B size rather than expanding 3×.

## The key insight

On a **flat surface**, a ternary mask only produces 3 rigid levels. On a **4-bit bumped terrain**, the same 1–1.58 bit mask produces **dozens of distinct, non-uniform effective values** because high hills amplify ternary decisions while valleys attenuate or zero them out.

## Why it matters

A dense `9B × (4 + 1 + 1)`-bit TerrainLoop model occupies **6.75 GB** for weights (**8× smaller** than 27B FP16, **2× smaller** than 27B INT4), avoids from-scratch ternary collapse, and executes 27B-equivalent sequential depth.
