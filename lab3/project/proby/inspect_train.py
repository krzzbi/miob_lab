import re, sys
sys.path.insert(0, "project")
from src.data_io import load_csv
from src.preprocess import mask_source_artifacts, tokenize_text

mids, texts, labels = load_csv("project/data/train.csv", include_label=True)
assert labels is not None

for tok in ("date", "url", "supplied", "sep"):
    s_hits = []
    h_hits = []
    for mid, t, l in zip(mids, texts, labels):
        m = mask_source_artifacts(t)
        toks = tokenize_text(t, apply_masking=True)
        if tok in toks:
            compact = re.sub(r"\s+", " ", m).strip()
            mm = re.search(rf"\b{tok}\b", compact, re.I)
            ctx = compact[max(0, mm.start()-40):min(len(compact), mm.end()+40)] if mm else compact[:80]
            (s_hits if l == "spam" else h_hits).append((mid, ctx))
    print(f"Token {tok!r}: spam={len(s_hits)}, ham={len(h_hits)}")
    for mid, ctx in h_hits[:3]:
        print(f"  [ham ] {mid}: ...{ctx}...")
    for mid, ctx in s_hits[:2]:
        print(f"  [spam] {mid}: ...{ctx}...")
