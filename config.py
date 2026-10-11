"""Paths, the donor classes, the models, and the corpus phrases. Every modelling choice
that is not code lives here.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# the sibling project whose etymology table and walk this one reuses
LOANWORD_DIR = os.environ.get("LOANWORD_DIR",
                              os.path.join(os.path.dirname(HERE), "loanword-flow-map"))
DATA = os.path.join(HERE, "data")
STATES = os.path.join(DATA, "states")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def flowmap():
    """languages.py and build_events.py from the loanword-flow-map project."""
    if LOANWORD_DIR not in sys.path:
        sys.path.insert(0, LOANWORD_DIR)
    import build_events
    import languages
    return languages, build_events


# Donor lineage (as languages.lineage names it) -> class. A class is a donor a reader
# would name: Anglo-Norman and Middle French are French, Middle Dutch and Middle Low
# German are one class because Wiktionary often cannot tell them apart ("from Middle
# Dutch or Middle Low German"). Old Norse is kept apart from the modern Scandinavian
# languages: the medieval loans (sky, they, egg) look native, the modern ones (ski,
# fjord, slalom) do not, and the hypothesis is about the first kind.
CLASS_OF = {
    "French": "French", "Norman": "French", "Picard": "French",
    "Latin": "Latin",
    "Greek": "Greek",
    "Old Norse": "Norse",
    "Dutch": "Dutch", "Low German": "Dutch", "Flemish": "Dutch",
    "German": "German", "Yiddish": "Yiddish",
    "Italian": "Italian",
    "Spanish": "Spanish",
    "Portuguese": "Portuguese",
    "Arabic": "Arabic",
    "Persian": "Persian",
    "Hebrew": "Hebrew",
    "Sanskrit": "Indic", "Hindi": "Indic", "Urdu": "Indic", "Hindustani": "Indic",
    "Bengali": "Indic", "Marathi": "Indic", "Gujarati": "Indic", "Punjabi": "Indic",
    "Prakrit": "Indic", "Pali": "Indic", "Tamil": "Dravidian", "Malayalam": "Dravidian",
    "Telugu": "Dravidian", "Kannada": "Dravidian",
    "Japanese": "Japanese",
    "Chinese": "Chinese",
    "Irish": "Celtic", "Scottish Gaelic": "Celtic", "Welsh": "Celtic", "Cornish": "Celtic",
    "Manx": "Celtic", "Breton": "Celtic", "Gaulish": "Celtic",
    "Russian": "Russian",
    "Turkish": "Turkish",
    "Malay": "Malay",
}
NATIVE = "native"
MIN_CLASS = 60          # words per class after all filters, or the class is dropped
MIN_ZIPF = 2.5          # wordfreq Zipf frequency: about 0.3 per million words

# Models. Pythia's suite shares one data order across sizes; OLMo 1 has an infini-gram
# index of its own corpus (Dolma 1.7).
PYTHIA = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
PYTHIA_STEPS = [0, 512, 2000, 8000, 32000, 143000]
OLMO = ["allenai/OLMo-1B-0724-hf", "allenai/OLMo-2-0425-1B"]

# The template every word is read in. Neutral on purpose: nothing about origin.
TEMPLATE = "Here is a word: {w}"

# infini-gram indexes of each model's training corpus
INDEX = {"pythia": "v4_piletrain_llama", "olmo1": "v4_dolma-v1_7_llama"}

# Phrases that mark an etymology claim about class c. Lineage names with a period prefix
# are the most specific; the bare "from X" forms also catch "translated from Arabic".
PHRASES = {
    "French": ["from Old French", "from Middle French", "from Anglo-Norman", "from French"],
    "Latin": ["from Latin", "from Medieval Latin", "from Late Latin", "from New Latin"],
    "Greek": ["from Greek", "from Ancient Greek"],
    "Norse": ["from Old Norse", "from Norse", "from Old Icelandic"],
    "Dutch": ["from Dutch", "from Middle Dutch", "from Middle Low German", "from Low German"],
    "German": ["from German", "from Middle High German"],
    "Yiddish": ["from Yiddish"],
    "Italian": ["from Italian"],
    "Spanish": ["from Spanish"],
    "Portuguese": ["from Portuguese"],
    "Arabic": ["from Arabic"],
    "Persian": ["from Persian"],
    "Hebrew": ["from Hebrew"],
    "Indic": ["from Sanskrit", "from Hindi", "from Urdu", "from Hindustani"],
    "Dravidian": ["from Tamil", "from Malayalam", "from Telugu"],
    "Japanese": ["from Japanese"],
    "Chinese": ["from Chinese", "from Cantonese", "from Mandarin"],
    "Celtic": ["from Irish", "from Scottish Gaelic", "from Welsh", "from Gaelic"],
    "Russian": ["from Russian"],
    "Turkish": ["from Turkish", "from Ottoman Turkish"],
    "Malay": ["from Malay"],
    NATIVE: ["from Old English", "from Middle English"],
}
# any etymology talk at all, whatever the donor
GENERIC = ["etymology", "etymologically", "loanword", "borrowed from"]   # the API allows 4 per clause
WINDOW = 50   # tokens: how far apart a word and a phrase may be to count as co-mentioned
