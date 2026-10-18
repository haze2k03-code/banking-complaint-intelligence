"""Regex patterns and text cleaning shared by the extraction and modelling steps."""
import re

# CFPB writes amounts like "{$1,200.00}" but consumers also type "$500" or "$ 75".
AMOUNT_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")

# CFPB masks personal dates as XX/XX/XXXX (sometimes XX/XX/2023).
MASKED_DATE_RE = re.compile(r"\b(?:XX|\d{1,2})/(?:XX|\d{1,2})/(?:XXXX|\d{2,4})\b")

FRAUD_RE = re.compile(
    r"\b(fraud\w*|identity theft|stolen identity|unauthori[sz]ed|scam\w*|"
    r"(?:did not|didn'?t|never) authori[sz]e\w*)\b",
    re.IGNORECASE,
)
FEE_RE = re.compile(r"\b(late fee|overdraft fee|annual fee|nsf fee|fees?|service charge)\b", re.IGNORECASE)
LEGAL_RE = re.compile(r"\b(lawyer|attorney|lawsuit|sue|suing|court|legal action)\b", re.IGNORECASE)

# Tokens to strip before modelling: redaction masks, braces, amounts, dates.
_MASK_RE = re.compile(r"\bX{2,}\b|\{|\}")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_SPACES_RE = re.compile(r"\s+")


def parse_amounts(text: str) -> list[float]:
    amounts = []
    for whole, cents in AMOUNT_RE.findall(text or ""):
        value = float(whole.replace(",", ""))
        if cents:
            value += float(f"0.{cents}")
        amounts.append(value)
    return amounts


def clean_text(text: str) -> str:
    """Lowercase, drop redaction masks/amounts/dates/punctuation for TF-IDF."""
    text = text or ""
    text = MASKED_DATE_RE.sub(" ", text)
    text = AMOUNT_RE.sub(" ", text)
    text = _MASK_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text.lower())
    return _SPACES_RE.sub(" ", text).strip()
