# Design Document

> **Status: draft.** This records the design and the evidence behind it so far. Results will be added once the models are trained and evaluated on the held-out test set.

## 1. Goal

Convert degraded smartphone RAW into high-quality sRGB, while improving colour accuracy, noise reduction and sharpness under mobile compute limits.

## 2. Key Decisions

### 2.1 Residual refinement instead of end-to-end rendering

**Decision:** The network refines a classical render.

**Why:** The classical stages already encode sensor physics and cost no parameters. Refinement should need a smaller network, and colour stays anchored to a real render. The learned arm is kept to test this assumption directly.

### 2.2 One NumPy implementation of the classical stages

**Decision:** No OpenCV or Torch in the classical branch.

**Why:** The same code runs in the data loader and the serverless function, which rules out train/serve skew and keeps the deployment small. The cost is some speed compared with optimised native code.

### 2.3 Noise measured from the mosaic

**Decision:** Fit $\sigma^2 = aI + b$ per capture from the two green planes.

**Why:** Test blocks carry no capture metadata, and a blind measurement works for any input. The green-plane difference cancels most image structure.

### 2.4 Known digital gain, not auto-exposure

**Decision:** Exposure compensation is an input.

**Why:** On training crops, linear signal maps to the target about 1:1, so references keep dark scenes dark. Blind auto-exposure would brighten those scenes and lose fidelity. Real phones know their shutter and ISO, so the input is realistic. The missing-metadata case is scheduled as a failure test.

### 2.5 Fitted colour matrices

**Decision:** Fit one matrix per camera by least squares on training scenes.

**Why:** Matrices derived from DNG `ColorMatrix2` showed a consistent bias of about +19% in the red ratio and -11% in the blue ratio across cameras.

### 2.6 White balance with a camera prior

**Decision:** Shrink shades-of-grey estimates towards a per-camera prior in log space, with weight 0.8.

**Why:** White balance is the largest classical error. On sampled training crops, as-shot white balance gave 37.1 dB and ΔE76 3.3, and plain shades-of-grey gave 30.2 dB and ΔE76 9.5. The prior reduced mean illuminant error from 7.3° to 4.2°. Crops are small, so one object can dominate the estimate.

**Caveat:** The prior comes from the dataset's illuminants, so it may not transfer to other cameras or lighting.

### 2.7 Two routed models

**Decision:** A small int8 student handles typical captures, and a larger fp16 teacher handles noisy ones.

**Why:** Most captures should not need the larger model. The gate is provisional, at `0.02` post-gain noise, and will be set on validation data.

### 2.8 Quantization boundaries

**Decision:** The output head and global colour head stay in floating point.

**Why:** The output adds directly to the render, so a coarsely quantized residual would appear as banding.

### 2.9 MongoDB is optional

**Decision:** Database calls have short timeouts and never block processing.

**Why:** Telemetry must not reduce availability of the core feature.

## 3. Data

See [DATASETS.md](DATASETS.md). The official SIDD server was unreachable, so the project uses SIDD-CC.

Known effects:

- Training crops are 256 × 256, so the network sees little global context for colour.
- One capture per scene is used, which limits disk use on the training machine.
- Tuning for white balance sampled cached scenes without excluding validation scenes. This affects validation slightly. It does not touch the test split.

## 4. Training

| Run | Width | Iterations | Notes |
|---|---:|---:|---|
| Teacher | 24 | 5,000 | Hybrid, quality route |
| Learned | 24 | 5,000 | Comparison arm |
| Student | 12 | 4,000 | Distilled from the teacher |
| Ablation variants | 12 | 2,000 | One change each |

These budgets reflect CPU-only training on a laptop. They are planned values.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Small crops limit global colour correction | Report per-camera results and analyse failures |
| Quantization harms quality | Compare fp32 and int8 on the same blocks |
| Short CPU training underfits | Keep the classical base, so the starting point is already reasonable |
| Results depend on one dataset | State this limitation, and do not generalise beyond it |

## 6. Results

Pending.
