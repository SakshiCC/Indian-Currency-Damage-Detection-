# Currency Note Recognition & Damage Detection - Integration Module

This module provides the independent, top-level integration layer connecting team member contributions without altering any teammate's private code or trained models.

---

## 1. Team Responsibilities & Ownership

| Member | Domain | Module Location | Status |
|---|---|---|---|
| **Yash** | Denomination Recognition (EfficientNetB0) | `src/`, `models/`, `results/` | Integrated via `yash_adapter.py` (read-only) |
| **Vinit** | Note Preprocessing & V3.1 Multi-Label Damage Classification | `vinit-damage-module/clip_damage_classifier/` | Integrated via `vinit_adapter.py` |
| **Sakshi** | Damage Segmentation, Mask Localization, Severity Assessment | Future Module | Interface hook exposed via `sakshi_interface.py` |

---

## 2. Zero-Modification Architecture

To respect team ownership and avoid merge conflicts:
- **`src/`, `models/`, and `results/` are NEVER modified.**
- All Yash imports are performed through `integration/yash_adapter.py`, which safely delegates to `src.denomination_predictor.predict_denomination()`.
- If TensorFlow is installed, Yash's real model executes. If TensorFlow is absent, the adapter provides a graceful, structured non-crashing status.
- Vinit's production code resides in `vinit-damage-module/clip_damage_classifier/`. Consumers import through `integration.vinit_adapter.VinitDamageAdapter`, preserving the V3.1 rules (`Burnt=1 ==> Stain=1`, `Normal=1 iff Burnt=0 AND Torn=0`).
- Sakshi's interface provides `get_damage_labels(result)` and `get_active_damage_labels(result)`, returning only physical damage labels (`["Torn", "Folded", "Burnt", "Stain"]`) strictly excluding `Normal`.

---

## 3. How to Run Inferences

### A. Denomination-Only Inference (Yash Module):
```python
from integration import predict_denomination_safe

result = predict_denomination_safe("path/to/banknote.jpg")
print("Denomination:", result["denomination"])
print("Confidence:", result["confidence_percent"], "%")
```

### B. Damage-Only Inference (Vinit V3.1 Module):
```python
from integration import predict_damage_safe

result = predict_damage_safe("path/to/banknote.jpg")
print("Final Damage Labels:", result["final_damage_labels"])
print("Predicted Active Labels:", result["predicted_labels"])
print("Classification Status:", result["classification_status"])
```

### C. Combined Full Pipeline Inference:
```python
from integration import run_currency_analysis

result = run_currency_analysis("path/to/banknote.jpg")
print("Denomination:", result["denomination"]["value"])
print("Damage Labels:", result["damage"]["final_labels"])
print("Sakshi Active Damages:", result["sakshi_active_damages"])
```

---

## 4. Combined Output Schema

```json
{
  "success": true,
  "image": {
    "path": "path/to/banknote.jpg",
    "filename": "banknote.jpg"
  },
  "denomination": {
    "value": "200",
    "confidence": 0.9812,
    "confidence_percent": 98.12,
    "class_index": 3,
    "is_background": false,
    "accepted": true,
    "available": true
  },
  "damage": {
    "raw_labels": {
      "Torn": 1,
      "Folded": 0,
      "Burnt": 0,
      "Stain": 1
    },
    "final_labels": {
      "Torn": 1,
      "Folded": 0,
      "Burnt": 0,
      "Stain": 1,
      "Normal": 0
    },
    "predicted_labels": ["Torn", "Stain"],
    "evidence_scores": {
      "Torn": 0.0063,
      "Folded": -0.0022,
      "Burnt": -0.0015,
      "Stain": -0.0012
    },
    "classification_status": "strong_prediction",
    "calibration_version": "v3.1",
    "normal_rule": "NOT(Burnt OR Torn)",
    "burnt_implies_stain": true
  },
  "preprocessing": {
    "note_detected": true,
    "perspective_applied": true
  },
  "sakshi_active_damages": ["Torn", "Stain"],
  "sakshi_handoff": {
    "image_path": "path/to/banknote.jpg",
    "is_damaged": true,
    "active_damage_labels": ["Torn", "Stain"]
  }
}
```

---

## 5. How Sakshi Can Consume Damage Labels

Sakshi's segmentation and severity module can consume Vinit's damage classification cleanly:

```python
from integration import get_damage_labels, get_active_damage_labels, get_sakshi_handoff_payload

# Option 1: Get full 5-class binary dictionary
labels_dict = get_damage_labels(result)
# -> {'Torn': 1, 'Folded': 0, 'Burnt': 0, 'Stain': 1, 'Normal': 0}

# Option 2: Get active physical damages only (strictly excludes Normal)
active_damages = get_active_damage_labels(result)
# -> ['Torn', 'Stain']

# Option 3: Full structured handoff payload
handoff = get_sakshi_handoff_payload(result, image_path="note.jpg")
if handoff["is_damaged"]:
    for damage_class in handoff["active_damage_labels"]:
        # Run Sakshi segmentation model on localized note crop
        pass
```
