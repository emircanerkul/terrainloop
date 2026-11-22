window.TERRAINLOOP_PAPER_HTML = `
<h1>TerrainLoop: Bumped-Terrain 4D Looped Compaction for 27B → 9B Language Models</h1>

<blockquote>
  <strong>Core thesis:</strong> Unrolling a shared 9B 4-bit terrain backbone across three normalized loops with terrain-coupled 1-bit/ternary modulators and cross-loop KV sharing preserves fine-grained weight geometry while delivering 27B-equivalent depth at ~6.75 GB of weight storage.
</blockquote>

<h2>Abstract</h2>
<p>Deep transformer deployments are bottlenecked by parameter storage, memory-bandwidth traffic during autoregressive decoding, and KV-cache growth across depth. Existing compression paradigms—uniform INT4 quantization, low-rank adaptation, structural pruning, and flat 1-bit or 1.58-bit ternary networks—treat each layer as an independent 3D slice or flatten all weights onto uniform low-bit plateaus. This paper proposes <strong>TerrainLoop</strong>, a 4D looped-stack architecture that unifies parameter sharing across layer depth <em>i</em> with <strong>terrain-coupled mixed precision</strong> (<code>4-bit + 1-bit/ternary + 1-bit/ternary</code>).</p>
<p>Instead of training three independent 9B blocks (27B total) or collapsing every layer to flat ternary levels <code>{-1, 0, +1}</code>, TerrainLoop stores a single <strong>9B 4-bit base backbone (H₀)</strong> that acts as a 16-level <em>bumped terrain</em>. Subsequent looped passes (<code>i = 1, 2</code>) apply deterministic zero-bit 4D rotations <code>D_i(H₀)</code> combined with lightweight <strong>1-bit or ternary masks (T₁, T₂)</strong> whose effective magnitudes are amplified by the local hill elevation <code>H₀</code> and local terrain slope <code>∇H₀</code>, stabilized by per-loop RMS normalization. Together with cross-loop KV sharing, a 3-loop TerrainLoop model delivers <strong>27B-equivalent sequential compute depth</strong> in <strong>6.75 GB</strong> of weight memory (or ~4.95 GB with 4D-factored masks) while remaining warm-startable from pretrained 9B checkpoints.</p>

<h2>1. Introduction</h2>
<p>Standard dense transformers store separate parameter matrices <code>W⁽⁰⁾, W⁽¹⁾, …, W⁽ᴸ⁻¹⁾</code> across <em>L</em> layers. Geometrically, each layer is a slice along a <strong>4th dimension — depth i</strong>. Treating consecutive slices as statistically independent forces a 27B model to allocate 54 GB in FP16 or 13.5 GB in INT4 even though adjacent layers share strong structural priors.</p>
<pre><code>3D slice view:   W⁽⁰⁾, W⁽¹⁾, W⁽²⁾ stored independently  →  3× storage, 3× bandwidth
4D looped view:  H₀ (4-bit hills) + D_i(H₀) + T_i (1-bit/ternary)  →  1× core + lightweight modulators</code></pre>
<p>Pure 1-bit (<code>{-1, +1}</code>) and ternary (<code>{-1, 0, +1}</code>, 1.58-bit) models reduce storage aggressively, but impose a <strong>flat terrain</strong>: every non-zero weight has identical scale <code>±α</code>. That flat geometry crushes outlier channels, loses subtle sensitivity ratios, and typically requires training from scratch over trillions of tokens. TerrainLoop separates responsibilities across depth: Loop 0 carries 4-bit precision and outliers, while Loops 1 and 2 carry low-bit functional divergence.</p>

<h2>2. Looped = Stacked Across the 4th Dimension</h2>
<p>Unrolling a shared transformer core <code>K = 3</code> times (<code>i ∈ {0, 1, 2}</code>) with iteration-conditioned weight transformations is functionally equivalent to a 3-fold stacked network obeying a compact 4D generative law:</p>

<div class="paper-diagram">
  <svg viewBox="0 0 880 230" width="820" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="tlArr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#d65c47"/>
      </marker>
    </defs>
    <rect x="16" y="74" width="125" height="62" rx="4" fill="#dce8df" stroke="#237d73" stroke-width="1.5"/>
    <text x="78" y="101" text-anchor="middle" font-size="13" font-weight="700" fill="#102b33">Input h₀</text>
    <text x="78" y="119" text-anchor="middle" font-size="11" fill="#38504f">token state</text>

    <rect x="180" y="64" width="180" height="82" rx="4" fill="#dce8df" stroke="#237d73" stroke-width="1.5"/>
    <text x="270" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#102b33">Loop 0 (i = 0)</text>
    <text x="270" y="110" text-anchor="middle" font-size="11" fill="#38504f">4-bit Terrain H₀ (9B)</text>
    <text x="270" y="128" text-anchor="middle" font-size="11" fill="#177b72">16 elevation levels</text>

    <rect x="400" y="64" width="195" height="82" rx="4" fill="#e7eee7" stroke="#237d73" stroke-width="1.5"/>
    <text x="497" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#102b33">Loop 1 (i = 1)</text>
    <text x="497" y="110" text-anchor="middle" font-size="11" fill="#38504f">D₁(H₀) + γ₁(H₀⊙T₁) + β₁∇H₀⊙T₁</text>
    <text x="497" y="128" text-anchor="middle" font-size="11" fill="#d65c47">RMSMatch + KV share</text>

    <rect x="635" y="64" width="195" height="82" rx="4" fill="#a7efcf" stroke="#237d73" stroke-width="1.8"/>
    <text x="732" y="90" text-anchor="middle" font-size="13" font-weight="700" fill="#09211f">Loop 2 (i = 2)</text>
    <text x="732" y="110" text-anchor="middle" font-size="11" fill="#28544d">D₂(H₀) + γ₂(H₀⊙T₂) + β₂∇H₀⊙T₂</text>
    <text x="732" y="128" text-anchor="middle" font-size="11" fill="#ba4b3b">27B Effective Depth</text>

    <line x1="141" y1="105" x2="176" y2="105" stroke="#d65c47" stroke-width="1.8" marker-end="url(#tlArr)"/>
    <line x1="360" y1="105" x2="396" y2="105" stroke="#d65c47" stroke-width="1.8" marker-end="url(#tlArr)"/>
    <line x1="595" y1="105" x2="631" y2="105" stroke="#d65c47" stroke-width="1.8" marker-end="url(#tlArr)"/>

    <path d="M 270 64 C 310 18, 450 18, 497 61" fill="none" stroke="#237d73" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#tlArr)"/>
    <path d="M 270 64 C 340 6, 660 6, 732 61" fill="none" stroke="#237d73" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#tlArr)"/>
    <text x="500" y="18" text-anchor="middle" font-size="11" fill="#177b72">Shared 4-bit terrain H₀ feeds Loops 1 &amp; 2 (zero weight duplication)</text>

    <path d="M 732 146 C 650 202, 340 202, 272 149" fill="none" stroke="#d65c47" stroke-width="1.6" stroke-dasharray="4,4" marker-end="url(#tlArr)"/>
    <text x="500" y="212" text-anchor="middle" font-size="11" fill="#d65c47">Backward STE: gradients from all 3 loops sum into H₀ while T₁, T₂ specialize</text>
  </svg>
</div>

<p>A naive linear trajectory such as <code>P(i) = P(0) + i·(1, 1, 2)</code> fails for real networks: norms diverge linearly (<code>3.74 → 6.13 → 8.54</code>), every step shifts along the same line, and arbitrary floats do not lie on a 4-bit grid. Using a quantized 4-bit anchor <code>H₀ = 0.25 × (+5, -3, +6) = (+1.25, -0.75, +1.50)</code> (<code>‖H₀‖ = 2.092</code>), norm-preserving 3D rotations <code>D_i</code>, terrain-scaled ternary steps (<code>γ = 0.40</code>), and RMS matching gives:</p>

<table>
  <thead>
    <tr><th>Loop <em>i</em></th><th>Deterministic Rotation D_i(H₀)</th><th>Ternary Mask T_i</th><th>Terrain Step 0.40(H₀ ⊙ T_i)</th><th>Normalized W(i)</th><th>Norm ‖W(i)‖₂</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>i = 0 (4-bit Base)</strong></td><td><code>(+1.25, -0.75, +1.50)</code></td><td>—</td><td>—</td><td><code>(+1.25, -0.75, +1.50)</code></td><td><strong>2.092</strong></td></tr>
    <tr><td><strong>i = 1 (Rot + T₁)</strong></td><td><code>(+0.10, -0.75, +1.95)</code></td><td><code>(+1, 0, -1)</code></td><td><code>(+0.50, 0.00, -0.60)</code></td><td><code>(+0.76, -0.95, +1.70)</code></td><td><strong>2.092</strong></td></tr>
    <tr><td><strong>i = 2 (Rot + T₂)</strong></td><td><code>(+0.10, -1.77, +1.11)</code></td><td><code>(-1, +1, +1)</code></td><td><code>(-0.50, -0.30, +0.60)</code></td><td><code>(-0.31, -1.60, +1.32)</code></td><td><strong>2.092</strong></td></tr>
  </tbody>
</table>

<h2>3. How TerrainLoop Differs from a Standard Looped Transformer</h2>
<p>A standard <strong>Looped Transformer</strong> reduces memory to 2 compact blocks (e.g., Attention + MLP) and reuses those exact same matrices identically across <code>× N</code> passes. While memory stays small, every iteration executes the same static function, limiting how much Loop 2 or Loop 3 can specialize.</p>
<p><strong>TerrainLoop</strong> keeps the 2 shared 4-bit terrain boxes (<code>H₀</code>) and attaches <strong>4 small 1-bit/ternary modifier chips</strong> so each pass activates its own lightweight modifier over the shared hills. Memory grows only slightly (2 base blocks + 4 thin modifier strips), yet every loop computes a distinct operator.</p>

<h3>Why Sparse Diffs Carry Huge Semantic &amp; Weight Power</h3>
<p>Consider representing four distinct statements using one base anchor plus tiny <code>+ / 0 / -</code> diffs:</p>
<table>
  <thead>
    <tr><th>Run / Pass</th><th>Slot 1 (Entity)</th><th>Slot 2 (Polarity)</th><th>Slot 3 (Domain)</th><th>Resulting Meaning</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Base (4-bit)</strong></td><td><code>App</code></td><td><code>is</code></td><td><code>software</code></td><td><strong>“App is software”</strong></td></tr>
    <tr><td><strong>Run 1 (Chip 1)</strong></td><td><code>+le</code></td><td><code>+n't</code></td><td><code>0 (skip)</code></td><td><strong>“Apple isn't software”</strong></td></tr>
    <tr><td><strong>Run 2 (Chip 2)</strong></td><td><code>+t</code></td><td><code>-n't</code></td><td><code>0 (skip)</code></td><td><strong>“Applet is software”</strong></td></tr>
    <tr><td><strong>Run 3 (Chip 3)</strong></td><td><code>0 (skip)</code></td><td><code>0 (skip)</code></td><td><code>+_layer</code></td><td><strong>“Applet is software layer”</strong></td></tr>
  </tbody>
</table>
<p>Notice the role of <code>0</code>: unchanged concepts like <code>software</code> cost zero update energy, while tiny positive/negative diffs flip both entity identity and truth polarity. In transformer residual layers, coupling 4-bit hills (16 levels) with ternary masks (3 states) yields <code>16 × 3 = 48</code> effective levels per loop vs. 3 flat levels in BitNet.</p>

<h2>4. Flat Surface vs. Bumped Terrain</h2>
<p>On a <strong>flat surface</strong> (BitNet b1.58 or pure 1-bit), every weight position sits at an identical scale plateau <code>α₀</code>. Adding <code>T₁, T₂ ∈ {-1, 0, +1}</code> creates only 3–5 rigid levels; tiny noisy weights and crucial outlier channels receive the exact same step size.</p>
<p>On a <strong>4-bit bumped terrain</strong> <code>H₀</code>, multiplying <code>T_i</code> by the local elevation <code>H₀</code> and local slope <code>∇_i H₀</code> turns each 1–1.58 bit mask into a terrain-aware modulator: high hills amplify ternary steps into large expressive shifts, valleys attenuate them into subtle refinements (or <code>0</code> skips them), and slopes push weights into orthogonal directions across loops. The result is <strong>30–80+ effective weight levels</strong> instead of 3 flat slabs.</p>

<h2>5. Mathematical Formulation &amp; Training</h2>
<pre><code>W_raw⁽ⁱ⁾ = D_i(H₀) + γ_i · (H₀ ⊙ T_i) + β_i · (∇_i H₀ ⊙ T_i)
W⁽ⁱ⁾     = W_raw⁽ⁱ⁾ · [ RMS(H₀) / (RMS(W_raw⁽ⁱ⁾) + ε) ]
h_{i+1}  = h_i + Block( RMSNorm_i(h_i) ; W⁽ⁱ⁾ )</code></pre>
<p>During backpropagation via the Straight-Through Estimator (STE), gradients from all three passes <strong>sum into the shared 4-bit terrain master weights</strong> (<code>∇_{H₀} L = Σ_i (∂L/∂W⁽ⁱ⁾)(∂W⁽ⁱ⁾/∂H₀)</code>), while <code>T₁, T₂</code> receive gradients proportional to <code>H₀</code> and <code>∇_i H₀</code>.</p>

<h2>6. Accounting for 27B → 9B Compaction</h2>
<table>
  <thead>
    <tr><th>Model Representation</th><th>Core Params</th><th>Precision Layout</th><th>Weight Memory</th><th>Runtime KV Cache</th><th>Effective Depth</th></tr>
  </thead>
  <tbody>
    <tr><td>Dense 27B (FP16)</td><td>27B</td><td>16-bit</td><td>54.00 GB</td><td>27B-layer KV</td><td>27B</td></tr>
    <tr><td>Dense 27B (INT4)</td><td>27B</td><td>4-bit</td><td>13.50 GB</td><td>27B-layer KV</td><td>27B</td></tr>
    <tr><td>Pure Ternary 27B (BitNet b1.58)</td><td>27B</td><td>1.58-bit packed</td><td>5.33 GB</td><td>27B-layer KV</td><td>27B</td></tr>
    <tr><td><strong>TerrainLoop 9B × 3 (Dense 1-bit)</strong></td><td><strong>9B core</strong></td><td><strong>4b + 1b + 1b</strong></td><td><strong>6.75 GB</strong></td><td><strong>9B-layer KV (shared)</strong></td><td><strong>27B</strong></td></tr>
    <tr><td><strong>TerrainLoop 9B × 3 (Dense Ternary)</strong></td><td><strong>9B core</strong></td><td><strong>4b + 1.58b + 1.58b</strong></td><td><strong>8.06 GB</strong></td><td><strong>9B-layer KV (shared)</strong></td><td><strong>27B</strong></td></tr>
    <tr><td><strong>TerrainLoop 9B × 3 (4D-Factored 20%)</strong></td><td><strong>~9B total</strong></td><td><strong>4b + factored 1b</strong></td><td><strong>4.95 GB</strong></td><td><strong>9B-layer KV (shared)</strong></td><td><strong>27B</strong></td></tr>
  </tbody>
</table>
<p>Note that group quantization scales and unlooped embedding/LM-head tables (~0.5–1.2 GB at 4–8 bit depending on vocabulary size) are accounted for separately from transformer block storage.</p>

<h2>7. Five Architectural Gaps &amp; Fixes</h2>
<table>
  <thead>
    <tr><th>Gap in Naive Looped Mixed-Precision</th><th>Failure Mode</th><th>TerrainLoop Fix</th></tr>
  </thead>
  <tbody>
    <tr><td>1. Additive/multiplicative variance growth</td><td>Weight variance scales as <code>(1 + γ₁² + γ₂²)σ²</code></td><td>Per-loop activation RMSNorm + <code>RMSMatch</code> weight rescaling</td></tr>
    <tr><td>2. 3× KV-cache expansion</td><td>Unrolling 9B layers 3× triples KV cache memory</td><td>Cross-loop KV sharing (Loops 1 &amp; 2 reuse Loop 0 / GQA KV)</td></tr>
    <tr><td>3. Same-subspace collapse</td><td>Elementwise scaling preserves eigenvectors</td><td>Deterministic 4D rotation &amp; slope <code>(D_i, ∇_i H₀)</code> per loop</td></tr>
    <tr><td>4. Valley sign-flip noise</td><td>Pure 1-bit forces near-zero weights to fire every loop</td><td>Ternary <code>{-1, 0, +1}</code> or terrain-gated 1-bit</td></tr>
    <tr><td>5. Coupled QAT oscillation</td><td>Moving 4-bit base and 1-bit masks together destabilizes STE</td><td>Warm-start <code>H₀</code> from pretrained 9B; anneal <code>T₁, T₂</code> first</td></tr>
  </tbody>
</table>

<h2>8. Evaluation Controls &amp; Roadmap</h2>
<p>TerrainLoop must be benchmarked against five controls: (1) Pretrained 9B INT4 single pass, (2) Looped 9B × 3 without modulators, (3) Flat 27B BitNet b1.58, (4) Dense 27B INT4, and (5) TerrainLoop 9B × 3. Metrics include perplexity, reasoning pass@1, general-task regression, weight bytes, peak VRAM with KV cache at 8k–32k context, and decoding tokens/sec.</p>
<p><strong>MVP Roadmap:</strong> Phase 1 stabilizes a 1.5B–3B warm-started prototype; Phase 2 benchmarks 9B × 3 against 27B INT4 and BitNet b1.58; Phase 3 implements 4D mask factorization and fused dequantize-modulate-matmul kernels.</p>

<h2>References</h2>
<ol>
  <li>Ma et al. <a href="https://arxiv.org/abs/2402.17764" target="_blank" rel="noreferrer">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>.</li>
  <li>Dehghani et al. <a href="https://arxiv.org/abs/1807.03819" target="_blank" rel="noreferrer">Universal Transformers</a>.</li>
  <li>Giannou et al. <a href="https://arxiv.org/abs/2301.13196" target="_blank" rel="noreferrer">Looped Transformers as Programmable Computers</a>.</li>
  <li>Lin et al. <a href="https://arxiv.org/abs/2306.00978" target="_blank" rel="noreferrer">AWQ: Activation-aware Weight Quantization</a>.</li>
  <li>Brandon et al. <a href="https://arxiv.org/abs/2405.12981" target="_blank" rel="noreferrer">Reducing Transformer Key-Value Cache Size with Cross-Layer Attention</a>.</li>
  <li>Dettmers et al. <a href="https://arxiv.org/abs/2305.14314" target="_blank" rel="noreferrer">QLoRA: Efficient Finetuning of Quantized LLMs</a>.</li>
</ol>
`;
