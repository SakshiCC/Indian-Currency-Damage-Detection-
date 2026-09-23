# currency-note-assessment-system
Deep learning based Indian currency note recognition and damage assessment system
# Currency Note Damage Assessment System

An image-processing and deep-learning system for automated **Indian currency note denomination recognition, physical damage detection, damage-area segmentation, and severity assessment**.

The system combines multiple computer-vision models and image-processing techniques into a single pipeline. A user provides an image of a currency note, and the system automatically identifies the denomination, detects possible physical damage, verifies and localizes the damaged regions, estimates the damaged area, and assigns a severity level.

---

## Features

- Indian currency denomination recognition
- Currency note preprocessing and rectification
- Multi-label physical damage detection
- Damage verification using image segmentation
- Torn/missing-area detection
- Fold/crease detection
- Stain detection
- Burnt-area detection support
- Multi-damage assessment
- Damaged-pixel calculation
- Estimated damage-area percentage
- Severity classification
- Recommendation generation
- Visual damage highlighting
- Integrated command-line interface
- Debug mode for model and segmentation analysis

---

## System Workflow

```text
Currency Note Image
        |
        v
Image Preprocessing / Rectification
        |
        v
Denomination Recognition
EfficientNetB0
        |
        v
Damage Type Classification
CLIP ViT-B/32
        |
        v
Damage Segmentation & Verification
OpenCV + Geometry Analysis
        |
        v
Final Damage Fusion
        |
        v
Combined Damage Mask
        |
        v
Damaged Pixel Calculation
        |
        v
Estimated Damage Area (%)
        |
        v
Severity Assessment
        |
        v
Recommendation
```

The damage-classification and segmentation stages complement each other. The trained damage classifier provides classification evidence about the possible type of damage, while the segmentation module verifies physical evidence and identifies the damaged regions.

---

# Main Modules

## 1. Denomination Recognition — EfficientNetB0

The denomination-recognition module identifies the value of the Indian currency note from the input image.

### Model

**EfficientNetB0 Transfer Learning Model**

### Main Processing

- Image loading
- Resizing
- Normalization
- Data augmentation during training
- Feature extraction using EfficientNetB0
- Denomination classification
- Confidence-score generation

### Output

Example:

```text
Denomination : Rs. 200
Confidence   : 99.55%
```

The predicted denomination is passed to the remaining stages of the assessment pipeline.

---

## 2. Currency Note Preprocessing

Before detailed damage analysis, the note image is prepared using image-processing operations.

Techniques used in the project include:

- Background handling
- Contour detection
- Note cropping
- Perspective correction
- Thresholding
- Morphological operations
- CLAHE
- Gaussian blur
- Median filtering
- Canny edge detection
- Adaptive thresholding
- Otsu thresholding
- Dilation
- Erosion

The purpose of this stage is to obtain a cleaner and more geometrically consistent representation of the currency note.

---

## 3. Damage Type Classification — CLIP ViT-B/32

Physical damage classification is performed using:

**OpenAI CLIP ViT-B/32**

The classifier analyzes the note and generates evidence for supported physical-damage categories.

### Supported Damage Categories

- Normal
- Torn
- Folded
- Stain
- Burnt

The module supports multi-label damage assessment because a currency note may contain more than one type of physical damage.

The trained damage-classification component achieved approximately **66% accuracy in the project evaluation**. Therefore, its predictions are passed through an additional segmentation-based verification stage before the final damage result is produced.

---

## 4. Damage Segmentation and Verification

The segmentation module determines **where physical damage occurs and how much of the note is affected**.

It combines OpenCV-based segmentation, note geometry, region analysis, connected-component filtering, and damage-specific processing.

### Torn Damage

Torn or missing material is estimated using the relationship between the expected complete note geometry and the visible note material.

Conceptually:

```text
Missing Material = Expected Note Region - Visible Note Region
```

The system maintains:

- Expected note mask
- Visible note mask
- Torn/missing-area mask
- Connected damaged regions

The final Torn mask is constrained to the estimated original note footprint to reduce background false positives.

### Folded Damage

Folded-note processing detects meaningful crease/fold evidence from the visible note area.

### Stain Damage

Stain segmentation identifies meaningful discoloration regions while constraining detections to visible note material.

### Burnt Damage

The system contains support for burnt-area segmentation. Positive validation for this category is more limited than for the primary tested damage categories and should therefore be interpreted conservatively.

---

## 5. Damage Verification and Fusion

The final damage type is not produced by blindly accepting a single model.

The system combines:

```text
Damage Classification Evidence
              +
Segmentation / Physical Evidence
              |
              v
      Final Damage Fusion
```

The classification model determines likely damage categories, while the segmentation module checks for meaningful physical evidence.

This helps the system:

- Verify predicted damage
- Reject unsupported classifications
- Recover strong physical damage missed by the classifier
- Handle multiple simultaneous damage types
- Prevent `Normal` from appearing together with confirmed physical damage

The user-facing result therefore presents a single field:

```text
Detected Damage Types : Torn, Stain
```

rather than exposing separate module predictions.

---

## 6. Damage Area Calculation

After verification, accepted physical-damage masks are combined.

For multiple damage types, the system uses the union of the masks so that overlapping damaged pixels are counted only once.

The estimated damage percentage is calculated as:

```text
Estimated Damage Area (%) =
(Damaged Pixels / Estimated Note Pixels) × 100
```

The percentage is obtained from the final segmentation mask.

Classification confidence or CLIP similarity scores are **not** used as the damage percentage.

---

## 7. Severity Assessment

The project uses experimental severity thresholds based on the estimated physical damage area.

| Estimated Damage Area | Severity |
|---:|---|
| 0% | None |
| > 0% to 5% | Minor |
| > 5% to 15% | Moderate |
| > 15% | Severe |

These thresholds are **project-defined assessment thresholds** and should not be interpreted as official Reserve Bank of India currency-fitness criteria.

---

## 8. Recommendation

The system generates a simple recommendation based on the final severity.

Examples:

```text
None:
No significant physical damage detected.

Minor:
Minor physical damage detected.

Moderate:
Moderate physical damage detected. Manual inspection recommended.

Severe:
Severe physical damage detected. Manual inspection recommended.
```

---

# Integrated Output

The final application presents the complete assessment as one result rather than exposing the internal modules separately.

Example:

```text
================================================================
              CURRENCY NOTE DAMAGE ASSESSMENT
================================================================

Input Image           : note.jpg

Denomination          : Rs. 200
Confidence            : 99.55%

Detected Damage Types : Torn, Stain

Damaged Pixels        : 37210
Estimated Note Pixels : 286608
Estimated Damage Area : 12.98%

Severity              : Moderate
Recommendation        : Moderate physical damage detected.
                        Manual inspection recommended.

Geometry Confidence   : 0.871

================================================================
```

Actual values depend on the supplied image.

---

# Damage Visualization

The system can generate a visual representation of the detected damaged regions.

The final visualization highlights accepted damaged areas using a **red overlay and/or red contour**.

The visualization is based on the same final verified damage mask used for:

- Damaged Pixels
- Estimated Damage Area
- Severity

This makes the result more interpretable by showing both **how much** damage was detected and **where** it was detected.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Denomination recognition |
| EfficientNetB0 | Currency denomination model |
| PyTorch | Deep-learning support |
| CLIP ViT-B/32 | Damage-type classification |
| OpenCV | Image processing and segmentation |
| NumPy | Image and numerical operations |
| Pillow | Image handling |
| Matplotlib | Result visualization |
| Hugging Face Transformers | CLIP model integration |
| Git / GitHub | Version control and collaboration |

---

# Project Structure

```text
currency-note-assessment-system/
│
├── run.py
├── requirements.txt
├── README.md
│
├── integration/
│   ├── currency_pipeline.py
│   ├── yash_adapter.py
│   ├── vinit_adapter.py
│   └── sakshi_interface.py
│
├── denomination/
│   └── ...
│
├── vinit-damage-module/
│   └── clip_damage_classifier/
│       └── ...
│
├── severity-module/
│   ├── config.py
│   ├── interface.py
│   ├── note_geometry.py
│   ├── pipeline_bridge.py
│   ├── recommendation.py
│   ├── region_analysis.py
│   ├── segmentation.py
│   ├── severity.py
│   ├── severity_assessor.py
│   ├── visualization.py
│   ├── run_all_tests.py
│   └── test_images/
│
├── test_final_verification.py
├── test_full_pipeline.py
├── test_integrated_runner_suite.py
├── test_real_adapters.py
├── test_true_e2e_pipeline.py
└── validate_all_damage_types.py
```

> The exact contents of model/data directories depend on the repository version and `.gitignore` configuration.

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd currency-note-assessment-system
```

## 2. Create a Python Environment

Using Conda:

```bash
conda create -n currency_note_env python=3.13
conda activate currency_note_env
```

Or use another compatible Python virtual environment.

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure the required trained model files are available at the paths expected by the project.

---

# Running the System

The easiest way to use the complete project is through the integrated runner.

```bash
python run.py
```

The application will ask:

```text
Enter currency note image path:
```

Provide the path to the currency-note image:

```text
Enter currency note image path: E:\Images\note.jpg
```

The complete processing pipeline then runs automatically.

---

## Direct Image Input

An image path can also be supplied directly:

```bash
python run.py "E:\Images\note.jpg"
```

Supported image formats include:

```text
.jpg
.jpeg
.png
```

---

# Debug Mode

For development and diagnostic information, run:

```bash
python run.py "E:\Images\note.jpg" --debug
```

Debug mode may expose additional internal information such as:

- Primary classifier labels
- Classification evidence
- Segmentation verification
- Added/removed damage labels
- Geometry information
- Mask statistics
- Reconstruction method

Debug information is intentionally hidden during normal execution to keep the final application output clean.

---

# Testing

Several tests are included for validating the individual modules and integrated pipeline.

### Final Verification

```bash
python test_final_verification.py
```

### True End-to-End Pipeline

```bash
python test_true_e2e_pipeline.py
```

### Integrated Runner Tests

```bash
python test_integrated_runner_suite.py
```

### Severity Module Tests

```bash
python severity-module/run_all_tests.py
```

The project includes checks for:

- Model loading
- Real-model execution
- Damage verification
- Normal exclusivity
- Multi-damage handling
- Damage-mask invariants
- Severity boundaries
- Region analysis
- Input safety
- Output consistency
- End-to-end module handoff

---

# Example Validation Results

Representative final integration tests produced results such as:

| Test Image | Denomination | Final Assessment | Estimated Damage Area | Severity |
|---|---:|---|---:|---|
| Torn Note | ₹10 | Torn + Stain | 13.02% | Moderate |
| Folded Note | ₹10 | Folded | 1.30% | Minor |
| Normal Note | ₹500 | Normal | 0.00% | None |
| Stained Note | ₹100 | Stain + Torn | 5.12% | Moderate |

These are results for the specific project test images and are **not general model-performance guarantees**.

---

# Important Design Principles

### Classification and Severity Are Separate

The damage classifier identifies possible damage categories.

The segmentation module determines the physical damaged area.

Therefore:

```text
Classification Confidence ≠ Damage Percentage
```

### Multiple Damage Types

A note may contain multiple damage types.

The final damage area is calculated using a combined mask so overlapping regions are counted only once.

### Normal Exclusivity

`Normal` cannot coexist with verified physical damage in the final result.

Valid:

```text
Detected Damage Types : Normal
```

or:

```text
Detected Damage Types : Torn, Stain
```

Invalid:

```text
Detected Damage Types : Normal, Torn
```

---

# Limitations

The current system has several limitations:

- Damage classification is not perfectly accurate.
- The damage-classification component achieved approximately 66% accuracy in project evaluation.
- Results can be affected by lighting conditions.
- Complex backgrounds can affect note extraction.
- Perspective and camera angle may influence geometry reconstruction.
- Severe occlusion may affect denomination recognition and segmentation.
- Fold and stain appearance can vary considerably between notes.
- Burnt-damage positive validation is currently limited.
- Segmentation estimates physical damage from a 2D image and should not be treated as exact physical measurement.
- Classification errors in upstream components may affect downstream processing, although the segmentation-verification layer is designed to reduce this effect.
- The project severity thresholds are experimental and are not official RBI note-fitness standards.

---

# Future Improvements

Possible future improvements include:

- Larger and more balanced damage datasets
- Improved multi-label damage classification
- Dedicated object-detection or segmentation models for individual damage categories
- More extensive burnt-note validation
- Improved note-boundary reconstruction
- Better handling of complex backgrounds
- Mobile-camera integration
- Real-time camera assessment
- Expanded denomination datasets
- Model optimization for faster inference
- Larger independent test-set evaluation

---

# Project Objective

The objective of this project is to demonstrate how **deep learning and classical image processing can be integrated into a single explainable pipeline for currency-note condition assessment**.

Rather than relying only on a classification label, the system combines denomination recognition, damage classification, segmentation, geometric analysis, severity estimation, and visual localization to provide a more informative assessment of a currency note.

---

## Disclaimer

This project is developed for **academic and research purposes**.

The generated damage percentage, severity level, and recommendation are project-specific estimates. They should not be interpreted as an official decision regarding the authenticity, legal validity, exchange eligibility, or fitness for circulation of any Indian currency note.
The generated damage percentage, severity level, and recommendation are project-specific estimates. They should not be interpreted as an official decision regarding the authenticity, legal validity, exchange eligibility, or fitness for circulation of any Indian currency note.
