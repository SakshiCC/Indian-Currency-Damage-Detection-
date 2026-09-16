"""
prompts.py
============================================================
Contrastive Positive and Negative Prompt Ensembles.
Each damage category features domain-adapted positive visual descriptions
and paired negative prompts that explicitly distinguish true damage from
visually confusing hard negatives (e.g. dark security ink, watermark shadows,
crease shadows, uneven photography borders, and normal currency patterns).
============================================================
"""

POSITIVE_PROMPTS = {
    "Torn": [
        "a close-up photograph showing a visible tear in an Indian banknote",
        "an Indian banknote with a physically ripped edge",
        "paper missing from the edge of a banknote",
        "an Indian banknote with a missing corner or missing piece",
        "an Indian currency note with a cut or torn section",
        "an Indian banknote with clearly visible tearing damage and separated paper",
        "an Indian rupee note that has torn paper or ripped borders"
    ],
    "Folded": [
        "a close-up photograph of an Indian banknote with heavy creases and folds",
        "an Indian banknote with a deep crease across the note",
        "an Indian banknote with visible physical fold marks",
        "a crumpled and severely wrinkled Indian currency note",
        "an Indian banknote with sharp paper creases across the portrait or center",
        "an Indian currency note with folding deformation and bent paper",
        "a bent and crumpled Indian rupee note with distinct crease lines"
    ],
    "Burnt": [
        "an Indian banknote with a charred blackened area from fire",
        "a photograph showing burnt edges and charred areas on an Indian banknote",
        "an Indian banknote with black scorch marks and heat char damage",
        "a fire-damaged Indian currency note with visible charring and singed holes",
        "an Indian banknote with partially burned paper and burnt holes",
        "an Indian rupee note with heat scorch marks and dark charred edges",
        "an Indian currency note with blackened burn marks from fire"
    ],
    "Stain": [
        "an Indian banknote with an ink stain",
        "a photograph of an Indian banknote with visible dark ink marks or writing",
        "an Indian banknote with noticeable oil or liquid stains",
        "an Indian currency note with dirty discoloration and smudges",
        "an Indian banknote with foreign colored marks, paint, or chemical spills",
        "a heavily soiled Indian currency note with visible surface stains",
        "an Indian rupee note with visible pen writing and ink smudges"
    ]
}

NEGATIVE_PROMPTS = {
    "Torn": [
        "a completely intact Indian banknote with no tears",
        "an Indian banknote with intact edges and no missing pieces",
        "an Indian currency note with complete borders and no ripped paper",
        "an Indian banknote with smooth, uncut and untorn edges",
        "a whole Indian rupee note with all four corners intact",
        "an Indian banknote with natural straight borders, not torn paper",
        "an Indian currency note showing lighting shadows at the border, not torn edges"
    ],
    "Folded": [
        "a completely flat and unwrinkled Indian banknote",
        "a smooth Indian currency note with no visible creases",
        "an Indian banknote with no fold marks and flat paper",
        "an Indian banknote that is crisp, flat, and unfolded",
        "a flat Indian rupee note without any crease lines",
        "an Indian banknote showing printed vertical lines and design elements, not fold creases",
        "an Indian currency note with flat surface without paper bending or wrinkles"
    ],
    "Burnt": [
        "an Indian banknote with no burn marks and no fire damage",
        "an Indian currency note with completely unburnt edges and paper",
        "an undamaged Indian banknote free from scorch marks or burns",
        "an Indian rupee note with clean paper and no charred areas",
        "an intact Indian currency note without any fire or heat damage",
        "an Indian banknote with dark printing, black serial numbers, and normal dark ink, not burnt",
        "an Indian currency note with camera shadows and dark background, not fire damage",
        "an Indian banknote with ink stains and dark soil, not heat scorched or burnt paper",
        "an Indian currency note with dark Mahatma Gandhi portrait printing, not charring"
    ],
    "Stain": [
        "a clean Indian banknote with no ink marks, writing, or stains",
        "an Indian currency note with a clean, unstained surface",
        "an Indian banknote free of discoloration, soil, and foreign marks",
        "a spotless Indian currency note without liquid or oil stains",
        "a pristine Indian rupee note with no pen marks or smudges",
        "an Indian banknote with normal currency printing, watermark, and seal, not a stain",
        "an Indian currency note with natural paper shading and background contrast, not dirty stains"
    ]
}


def get_positive_prompts(category: str):
    if category not in POSITIVE_PROMPTS:
        raise ValueError(f"Unknown category '{category}'")
    return POSITIVE_PROMPTS[category]


def get_negative_prompts(category: str):
    if category not in NEGATIVE_PROMPTS:
        raise ValueError(f"Unknown category '{category}'")
    return NEGATIVE_PROMPTS[category]
