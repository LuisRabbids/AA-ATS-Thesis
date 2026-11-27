"""
app.py
Interactive demo for "Anatomically-Guided Adaptive Masking for Masked Autoencoder
Pretraining in Brain MRI Tumour Segmentation".

Pick a patient, scroll through slices, and see what each pre-training strategy's
fine-tuned segmenter predicts, against the reference annotation, with Dice computed
live for the slice on screen.

Two models are shown side by side on purpose. On the experimental partition the
anatomically-guided model scored 0.64 Dice points below plain random masking, a
difference that is statistically reliable (it is several times the 0.11-point spread
measured between identical training runs) and at the same time invisible to the eye.
Showing both is the honest way to present that: the finding is real, and it is small.

Both models come from fold 3 of the three-fold evaluation, and the example patients are
drawn from that fold's validation set, so neither network has seen them in pre-training
or fine-tuning.

Run it with
    streamlit run app.py

Weights and example data are downloaded once from the repository's release page and
cached locally; nothing else is needed.
"""

import sys
import types
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.colors import ListedColormap             # noqa: E402

# finetune.py imports the training data layer, which pulls in OpenCV and pandas. The
# demo needs only the network definition, so the module is stubbed out beforehand.
if "data" not in sys.modules:
    _stub = types.ModuleType("data")
    _stub.BraTSSliceDataset = _stub.load_split = None
    sys.modules["data"] = _stub

from finetune import BACKBONES, REGIONS, SegViT          # noqa: E402

RELEASE = ("https://github.com/LuisRabbids/AA-ATS-Thesis/releases/download/v1.0-demo")
CACHE = Path.home() / ".cache" / "aa-ats-demo"
MODALITIES = ["T1", "T1Gd", "T2", "FLAIR"]
MODELS = {"ours": "Ours — anatomically guided", "mae": "MAE — random masking"}
CLASSES = {1: "Necrotic / non-enhancing core", 2: "Oedema", 3: "Enhancing tumour"}
COLOURS = {1: "#e4572e", 2: "#3fa7d6", 3: "#f2c14e"}
FOLD3_DSC = {"ours": 0.8456, "mae": 0.8512}              # whole fold, 350 patients


# ------------------------------------------------------------------ assets
def fetch(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with st.spinner(f"Downloading {name} (first run only)…"):
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"{RELEASE}/{name}", tmp)
            tmp.rename(path)
    return path


@st.cache_resource(show_spinner=False)
def load_model(key):
    ck = torch.load(fetch(f"{key}_f3.pt"), map_location="cpu")
    model = SegViT(**BACKBONES[ck.get("backbone", "small")])
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_cases():
    z = np.load(fetch("demo_cases.npz"))
    ids = sorted({k.rsplit("_", 1)[0] for k in z.files})
    return {c: {"img": z[f"{c}_img"], "seg": z[f"{c}_seg"]} for c in ids}


@st.cache_data(show_spinner=False)
def predict(key, case, idx, _img):
    """_img is passed with a leading underscore so Streamlit hashes the ids, not the array."""
    x = torch.from_numpy(_img.astype(np.float32) / 255.0)[None]   # matches the data layer
    with torch.no_grad():
        return load_model(key)(x).argmax(1)[0].numpy().astype(np.uint8)


# ------------------------------------------------------------------ metrics and drawing
def dice(pred, gt, labels):
    p, g = np.isin(pred, labels), np.isin(gt, labels)
    total = p.sum() + g.sum()
    return 1.0 if total == 0 else 2.0 * (p & g).sum() / total


def panel(background, labels, title):
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=140)
    fig.patch.set_alpha(0)
    ax.imshow(background, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if labels is not None:
        masked = np.ma.masked_where(labels == 0, labels)
        ax.imshow(masked, cmap=ListedColormap([COLOURS[1], COLOURS[2], COLOURS[3]]),
                  vmin=1, vmax=3, alpha=0.45, interpolation="nearest")
    ax.set_title(title, fontsize=9, color="#888", pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


def show(fig):
    """Render a figure and release it; Streamlit reruns the script on every widget
    change, so figures left open accumulate for the life of the session."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ------------------------------------------------------------------ page
st.set_page_config(page_title="Adaptive Masking — Tumour Segmentation",
                   layout="wide", page_icon="🧠")

st.title("Anatomically-guided adaptive masking")
st.caption("Brain tumour segmentation on BraTS 2021 · fold 3 · patients unseen in "
           "pre-training and fine-tuning")

cases = load_cases()

with st.sidebar:
    st.header("Controls")
    case = st.selectbox("Patient", list(cases), format_func=lambda c: c.replace("_", " "))
    n = len(cases[case]["seg"])
    idx = st.slider("Slice", 1, n, n // 2 + 1) - 1
    modality = st.selectbox("Background image", MODALITIES, index=3)
    chosen = st.multiselect("Models", list(MODELS), default=list(MODELS),
                            format_func=lambda k: MODELS[k])
    st.divider()
    st.caption("Colours")
    for lab, name in CLASSES.items():
        st.markdown(
            f"<span style='display:inline-block;width:11px;height:11px;border-radius:2px;"
            f"background:{COLOURS[lab]};margin-right:7px'></span>{name}",
            unsafe_allow_html=True)

img = cases[case]["img"][idx]
gt = cases[case]["seg"][idx]
background = img[MODALITIES.index(modality)].astype(np.float32) / 255.0

columns = st.columns(1 + len(chosen))
with columns[0]:
    show(panel(background, gt, "Reference annotation"))
    st.caption(f"{modality} · slice {idx + 1} of {n}")

rows = {}
for col, key in zip(columns[1:], chosen):
    pred = predict(key, case, idx, img)
    rows[key] = {r: dice(pred, gt, labs) for r, labs in REGIONS.items()}
    with col:
        show(panel(background, pred, MODELS[key]))
        st.caption(f"Dice this slice — " + " · ".join(
            f"{r} {rows[key][r]:.3f}" for r in REGIONS))

if rows:
    st.divider()
    left, right = st.columns([2, 3])
    with left:
        st.subheader("This slice")
        st.dataframe(
            {"Model": [MODELS[k] for k in rows],
             **{r: [f"{rows[k][r]:.3f}" for k in rows] for r in REGIONS}},
            hide_index=True, use_container_width=True)
    with right:
        st.subheader("Whole fold, 350 patients")
        st.dataframe(
            {"Model": [MODELS[k] for k in rows],
             "Dice (3D, mean of WT/TC/ET)": [f"{FOLD3_DSC[k]:.4f}" for k in rows]},
            hide_index=True, use_container_width=True)
        st.caption("Single slices vary far more than the models do. The fold-level "
                   "figures are the ones the thesis reports; per-slice Dice is shown "
                   "to make the overlay legible, not as a result.")

with st.expander("What am I looking at?"):
    st.markdown(f"""
Both models share an identical architecture, training schedule and fine-tuning
procedure. **They differ only in how patches were masked during self-supervised
pre-training** — the proposed method selects them from a learned, anatomy-informed
distribution, the baseline selects them uniformly at random.

Over the full three-fold evaluation (1,051 patients) the proposed method scored
**0.33 Dice points below** random masking. Three runs of an identical configuration
differ by only 0.11 points, so that gap is not training noise — but it is also far too
small to see in any single image, which is why both predictions look alike here.

The example patients come from fold 3's validation set, so neither network saw them
during pre-training or fine-tuning. Slices without tumour were dropped; what remains is
every fourth annotated slice containing a lesion.

Labels follow the BraTS convention: whole tumour (WT) is all three classes, tumour core
(TC) is necrotic plus enhancing, and enhancing tumour (ET) is the enhancing class alone.
""")
