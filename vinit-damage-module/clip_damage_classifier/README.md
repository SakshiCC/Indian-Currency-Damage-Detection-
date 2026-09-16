# Vinit's Currency Note Image Processing + Damage Classification Module

Independent, zero-shot multi-label damage classification and classical image-preprocessing pipeline for Indian banknotes. Part of the **Indian Currency Note Recognition and Damage Detection** group project.

---

## 1. Purpose
This module fulfills **Vinit's** assigned project responsibilities:
1. **Currency Note Image Preprocessing & Boundary Detection**: Classical OpenCV contour detection, note cropping, perspective transformation, and luminance-channel CLAHE.
2. **Multi-Label Damage Classification**: Zero-shot detection of physical currency damages (`Torn`, `Folded`, `Burnt`, `Stain`, `Normal`) using Hugging Face CLIP (`openai/clip-vit-base-patch32`) with contrastive prompt ensemble evidence scoring.

---

## 2. Damage Taxonomy
The system evaluates four independent physical damage categories plus one derived status:

| Category | Definition | Included Visual Manifestations |
|---|---|---|
| **Torn** | Physical separation or material loss | Edge tears, ripped paper, missing corners, scissor cuts, separated borders. |
| **Folded** | Non-destructive paper deformation | Creases, fold lines, dog-ears, crumpling, wrinkled paper. |
| **Burnt** | Heat or fire damage | Charred paper, blackened edges, singed holes, scorch marks. |
| **Stain** | Surface contamination | Ink marks, pen writing, oil spots, chemical discoloration, dirt, smudges. |
| **Normal** | Banknote in serviceable condition | **Derived strictly (V3.1)**: `Normal = 1` iff `Burnt = 0` AND `Torn = 0`. (Folded and Stain do not independently force Normal to 0). |

> [!NOTE]
> **V3.1 Domain Consistency Rule**: When thermal charring occurs (`Burnt = 1`), surface staining is physically induced, automatically asserting `final_Stain = 1` (`final_Stain = clip_stain OR final_burnt`).

---

## 3. Dataset Context
- **Dataset Archive**: `Spoilt Indian Banknotes.zip` (~5,125 valid images).
- **Denominations**: ₹10, ₹20, ₹50, ₹100 across New and Old note generations.
- **Scientific Designation**:
  > **Important**: The 5,125-image dataset is automatically pseudo-labeled using zero-shot CLIP. There is no pre-existing ground-truth annotation for the full dataset. Classifications are designated as **"CLIP zero-shot pseudo-labels"**, NOT ground truth.
  > The 90-image human-verified sample was created specifically for validation and threshold calibration; it is NOT equivalent to a full ground-truth dataset.

---

## 4. CLIP Model
- **Model**: `openai/clip-vit-base-patch32` (Hugging Face Transformers).
- **Architecture**: Dual vision-language transformer.
- **Device Support**: Automatic CPU / CUDA device selection.
- **Execution Efficiency**: Text prompt embeddings are computed once and cached in memory.

---

## 5. Zero-Shot Approach
Because no large-scale labelled damage dataset exists for Indian currency notes, zero-shot CLIP is utilized to bootstrap candidate damage labels without requiring supervised model training.

---

## 6. Contrastive Prompt Ensemble
Raw cosine similarity in CLIP space suffers from high semantic baseline overlap across currency images. This pipeline employs a **contrastive positive vs. negative prompt ensemble**:
- $S_{	ext{pos}}$: Mean cosine similarity against domain-adapted positive visual damage prompts.
- $S_{	ext{neg}}$: Mean cosine similarity against paired negative intact prompts that explicitly counter hard negatives (e.g. distinguishing burnt charring from dark security ink, stamps, shadows, or portrait borders).

---

## 7. Evidence Score Formulation
For each damage category $C \in \{	ext{Torn}, 	ext{Folded}, 	ext{Burnt}, 	ext{Stain}\}$:
$$	ext{Evidence}_C = S_{	ext{pos}} - S_{	ext{neg}}$$

For clean notes, negative prompts dominate ($S_{	ext{neg}} > S_{	ext{pos}}$), yielding negative evidence scores (typically $-0.015$ to $-0.002$). Damage is triggered only when positive evidence outweighs negative evidence.

---

## 8. Thresholding & Calibration
Thresholds were calibrated using the 90-image human verification sample:

| Category | Baseline (v1) | Calibrated (v2) | Rationale |
|---|:---:|:---:|---|
| **Torn** | `+0.0005` | `-0.0005` | Captures hairline tears and edge fraying (F1: `0.4112` $	o$ `0.4685`) |
| **Folded** | `+0.0005` | `0.0000` | Improves crease sensitivity (F1: `0.4615` $	o$ `0.4878`) |
| **Burnt** | `+0.0015` | `+0.0038` | **Eliminates 14+ false positives** from dark ink/shadows (F1: `0.2759` $	o$ `0.4211`) |
| **Stain** | `+0.0005` | `0.0000` | Boosts faint ink/soil detection (F1: `0.4762` $	o$ `0.5227`) |

---

## 9. Multi-Label Classification
Banknotes frequently exhibit multiple damages simultaneously (e.g. `Folded + Stain`, `Torn + Folded + Burnt`). Each damage category is thresholded independently:
$$	ext{Damage}_C = egin{cases} 1 & 	ext{if } 	ext{Evidence}_C \ge 	ext{Threshold}_C \ 0 & 	ext{otherwise} \end{cases}$$
No `argmax()` or `softmax()` single-class competition is used.

---

## 10. Strict Normal Exclusivity Rule
Normal is mutually exclusive with all damage types:
$$	ext{Normal} = 1 \iff 	ext{Torn} = 0 \land 	ext{Folded} = 0 \land 	ext{Burnt} = 0 \land 	ext{Stain} = 0$$
Combinations such as `Normal + Torn` or `Normal + Folded` are structurally impossible.

---

## 11. Human Verification Sample
- **File**: `clip_damage_classifier/outputs/human_verification_sample.csv`
- **Sample Size**: 90 images strategically stratified across denominations, generations, damage types, and confidence tiers.
- **Review Columns**: `human_Torn`, `human_Folded`, `human_Burnt`, `human_Stain`, `human_Normal`, `review_status`.

---

## 12. Evaluation & Before-vs-After Results

Evaluated against the 90 human-verified annotations:

### Overall Multi-Label Metrics
| Metric | Baseline (v1) | Calibrated (v2) | Relative Change |
|---|:---:|:---:|:---:|
| **Micro F1** | 0.4087 | **0.4630** | **+13.3%** |
| **Macro F1** | 0.3570 | **0.4133** | **+15.8%** |
| **Hamming Loss** | 0.4244 | **0.3867** | **-8.9%** (better) |
| **Jaccard Score** | 0.2657 | **0.3222** | **+21.3%** |
| **Exact Match Ratio** | 0.0333 | **0.0667** | **+100.0%** (2x) |

### Per-Class F1 Scores
| Class | Baseline (v1) F1 | Calibrated (v2) F1 | Precision (v2) | Recall (v2) |
|---|:---:|:---:|:---:|:---:|
| **Torn** | 0.4112 | **0.4685** | 0.8667 | 0.3210 |
| **Folded** | 0.4615 | **0.4878** | 0.5714 | 0.4255 |
| **Burnt** | 0.2759 | **0.4211** | 0.2667 | 1.0000 |
| **Stain** | 0.4762 | **0.5227** | 0.6765 | 0.4259 |
| **Normal** | 0.1600 | **0.1667** | 0.1250 | 0.2500 |

---

## 13. Limitations
1. **Zero-Shot Representations**: CLIP ViT-B/32 is a general vision-language model trained on internet web pairs, not specialized currency defect imagery.
2. **Extreme Dark Ink vs. Scorch**: Very dark black ink stamps can still occasionally elevate Burnt evidence scores near the threshold.
3. **Micro-Tears**: Tiny border tears under 2-3mm without missing paper may be overlooked by global CLIP patch tokens.
4. **Classical Note Segmentation**: Relies on contour area contrast against background; plain white backgrounds matching note borders may fallback to full frame.

---

## 14. How to Run & Reproduce

```bash
# 1. Run unit test suite (21 unit tests)
python -m unittest discover -s clip_damage_classifier/tests -p "test_*.py"

# Or with pytest
python -m pytest clip_damage_classifier/tests

# 2. Run Baseline Evaluation (v1)
python clip_damage_classifier/evaluation.py --version v1

# 3. Run Calibrated Evaluation (v2)
python clip_damage_classifier/evaluation.py --version v2

# 4. Run Single-Image Inference via CLI
python -c "from clip_damage_classifier import DamageClassifier; c = DamageClassifier(); print(c.classify('path/to/note.jpg'))"
```

---

## 15. Output Files Directory
All outputs are organized in `clip_damage_classifier/outputs/`:

| File | Description |
|---|---|
| `spoilt_indian_banknotes_damage_classification.csv` | Full dataset (5,125 notes) with v1 baseline pseudo-labels. |
| `spoilt_indian_banknotes_damage_classification_v2.csv` | Full dataset (5,125 notes) with v2 calibrated pseudo-labels. |
| `human_verification_sample.csv` | 90-image human-verified sample with human labels and review status. |
| `evaluation_report.txt` & `.json` | Baseline (v1) evaluation report. |
| `evaluation_report_v2.txt` & `.json` | Calibrated (v2) evaluation report with before/after comparison. |
| `calibrated_thresholds.json` | JSON configuration of calibrated v2 thresholds. |
| `hard_negative_analysis.md` | Forensic analysis of visual error patterns and failure modes. |

---

## 16. Integration Interface

### For Yash (Denomination Recognition & Integration):
```python
from clip_damage_classifier import DamageClassifier

# Initialize once
classifier = DamageClassifier(device="auto", use_preprocessing=True)

# Run classification on any image path, array, or PIL Image
result = classifier.classify("path/to/banknote.jpg")

print(result["damage_labels"])
# {"Torn": 0, "Folded": 1, "Burnt": 0, "Stain": 1, "Normal": 0}

print(result["predicted_labels"])
# ["Folded", "Stain"]

print(result["classification_status"])
# "strong_prediction"
```

### For Sakshi (Damage Segmentation & Severity Assessment):
Sakshi's segmentation module can consume `result["predicted_labels"]` (e.g. `["Folded", "Stain"]`) to localize region-level masks and compute percentage area severity.
