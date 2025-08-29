import argparse
import re
import unicodedata
from typing import List, Optional

import pandas as pd


# Compile regexes once for performance.
# Dimension patterns: capture triplets/pairs like 140x190 cm, 8x80x25cm, with unit conversion to cm.
NUMBER = r"(?:\d{1,4}(?:[\.,]\d{1,2})?)"
UNITS = r"(?:mm|cm|dm|m|in|pouce(?:s)?|pieds?|pied|ft)"
SEP = r"[xX×✕]"

# Triplet with optional per-number units and/or trailing unit
TRIPLE_DIM_RE = re.compile(
    fr"\b(?P<a>{NUMBER})\s*(?P<ua>{UNITS})?\s*{SEP}\s*(?P<b>{NUMBER})\s*(?P<ub>{UNITS})?\s*{SEP}\s*(?P<c>{NUMBER})\s*(?P<uc>{UNITS})?\s*(?P<ut>{UNITS})?\b",
    re.IGNORECASE,
)
# Pair with optional per-number units and/or trailing unit
PAIR_DIM_RE = re.compile(
    fr"\b(?P<a>{NUMBER})\s*(?P<ua>{UNITS})?\s*{SEP}\s*(?P<b>{NUMBER})\s*(?P<ub>{UNITS})?\s*(?P<ut>{UNITS})?\b",
    re.IGNORECASE,
)
# Single dimension requires a length unit to avoid plain numbers
SINGLE_DIM_RE = re.compile(
    fr"\b(?:(?:diamet(?:re|re)|diam(?:\.|etre)?|ø|hauteur|largeur|longueur|profondeur|e?paisseur)\s*)?(?P<a>{NUMBER})\s*(?P<ua>{UNITS})\b",
    re.IGNORECASE,
)

# Colors list (French + some English fallbacks). Accent-insensitive matching.
COLOR_WORDS: List[str] = [
    # base colors
    "noir", "blanc", "gris", "rouge", "rose", "vert", "bleu", "jaune", "orange", "violet", "marron", "beige", "taupe",
    # shades/phrases
    "gris clair", "gris foncé", "gris fonce", "bleu clair", "bleu foncé", "bleu fonce", "bleu marine", "bleu canard", "bleu turquoise",
    "rose pâle", "rose pale", "vert clair", "vert foncé", "vert fonce", "gris anthracite", "anthracite", "argent", "doré", "dore",
    # material/wood tones often used as colors
    "chêne", "chene", "chêne clair", "chene clair", "chêne foncé", "chene fonce", "noyer", "pin", "naturel",
    # others
    "camel", "bordeaux", "turquoise", "transparent", "translucide", "ivoire", "crème", "creme", "graphite", "multicolore", "multicolor",
]

def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")

# Build color regex on accentless, lowercased phrases
COLOR_WORDS_NORM = sorted({strip_accents(c.lower()) for c in COLOR_WORDS}, key=len, reverse=True)
COLOR_PATTERN_NORM = re.compile(r"\b(" + "|".join(map(re.escape, COLOR_WORDS_NORM)) + r")\b", re.IGNORECASE)

# Map normalized (no accent) color phrases back to a canonical accented form when possible
CANONICAL_COLOR_MAP = {}
for original in COLOR_WORDS:
    norm = strip_accents(original.lower())
    # Prefer accented original if multiple map to same norm
    if norm not in CANONICAL_COLOR_MAP or (any(ch in original for ch in "àâäéèêëîïôöùûüçñ") and not any(ch in CANONICAL_COLOR_MAP[norm] for ch in "àâäéèêëîïôöùûüçñ")):
        CANONICAL_COLOR_MAP[norm] = original.lower()


def normalize_number(num_str: str) -> str:
    # Convert French decimals (comma) to dot; remove thousands separators (space)
    s = num_str.replace(" ", "")
    s = s.replace(",", ".")
    return s


def parse_float(num_str: str) -> float:
    return float(normalize_number(num_str))


def convert_to_cm(value: float, unit: Optional[str]) -> float:
    if not unit:
        return value
    u = unit.lower()
    if u in ("cm",):
        return value
    if u in ("mm",):
        return value / 10.0
    if u in ("dm",):
        return value * 10.0
    if u in ("m",):
        return value * 100.0
    if u in ("in", "pouce", "pouces"):
        return value * 2.54
    if u in ("ft", "pied", "pieds"):
        return value * 30.48
    # Unknown unit: return as-is
    return value


def format_cm(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def extract_first_dimension(text: str) -> Optional[str]:
    if not text:
        return None
    # Try triple first
    m = TRIPLE_DIM_RE.search(text)
    if m:
        a = convert_to_cm(parse_float(m.group("a")), m.group("ua") or m.group("ut"))
        b = convert_to_cm(parse_float(m.group("b")), m.group("ub") or m.group("ut"))
        c = convert_to_cm(parse_float(m.group("c")), m.group("uc") or m.group("ut"))
        return f"{format_cm(a)} * {format_cm(b)} * {format_cm(c)}"
    # Then pair
    m = PAIR_DIM_RE.search(text)
    if m:
        a = convert_to_cm(parse_float(m.group("a")), m.group("ua") or m.group("ut"))
        b = convert_to_cm(parse_float(m.group("b")), m.group("ub") or m.group("ut"))
        return f"{format_cm(a)} * {format_cm(b)}"
    # Then single with unit only
    m = SINGLE_DIM_RE.search(text)
    if m:
        a = convert_to_cm(parse_float(m.group("a")), m.group("ua"))
        return f"{format_cm(a)}"
    return None


def extract_colors(text: str) -> Optional[str]:
    if not text:
        return None
    norm_text = strip_accents(text.lower())
    matches = [m.group(0).lower() for m in COLOR_PATTERN_NORM.finditer(norm_text)]
    if not matches:
        return None
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in matches:
        if c not in seen:
            seen.add(c)
            # Map back to canonical form with accents when available
            unique.append(CANONICAL_COLOR_MAP.get(c, c))
    return ", ".join(unique)


def process_csv(input_path: str, output_path: str, label_col: str = "Libellé produit") -> None:
    # Use a tolerant CSV reader setup to handle messy real-world data
    df = pd.read_csv(
        input_path,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        on_bad_lines="skip",
    )
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {input_path}. Columns: {list(df.columns)}")
    labels = df[label_col].astype(str)
    df["extracted_dimension"] = labels.apply(extract_first_dimension)
    df["extracted_colors"] = labels.apply(extract_colors)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dimensions and colors from product labels using regex")
    parser.add_argument("--input", required=True, help="Path to input CSV (must include 'Libellé produit' column)")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--label-col", default="Libellé produit", help="Name of the label column")
    args = parser.parse_args()
    process_csv(args.input, args.output, args.label_col)


if __name__ == "__main__":
    main()


