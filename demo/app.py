"""
OverflowGuard Live Inference Demo — Streamlit app.

Run with:  streamlit run app.py
"""

import gc
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
from bs4 import BeautifulSoup
from ddgs import DDGS

from mock_model import MockModel

try:  # heavy/optional: only needed for the (currently disabled) real-model path
    from routers import ROUTER_CLASSES
except Exception:
    ROUTER_CLASSES = {}

st.set_page_config(
    page_title="OverflowGuard Demo", layout="wide", initial_sidebar_state="expanded"
)

# ── Constants ────────────────────────────────────────────────────────
MODELS = ["xRAG", "PISCO", "OSCAR"]
COLORS = {"xRAG": "#5ED1FF", "PISCO": "#8238D9", "OSCAR": "#FFAF5E"}
MODEL_PATH = {
    "PISCO": "wexumin/pisco-7b-router",
    "OSCAR": "wexumin/oscar-7b-router",
    "xRAG": "wexumin/xrag-7b-router",
}
TOKENS_PER_CHUNK = {"xRAG": 1, "PISCO": 8, "OSCAR": 8}

# ── Model source ─────────────────────────────────────────────────────
# Flip this to True (or run `OG_USE_REAL=1 streamlit run app.py`) to load the
# real routers from Hugging Face; default False → fast, dependency-free mock.
USE_REAL_MODELS = os.environ.get("OG_USE_REAL", "0").lower() in ("1", "true", "yes")

_MODEL_KEY_MAP = {"pisco": "PISCO", "oscar": "OSCAR", "xrag": "xRAG"}


def _load_presets():
    """Load new_presets.jsonl (one preset per line), normalized to the shape the
    rest of the app uses: {key: {context, query(str), gold(list), results{DISP: ...}}}."""
    path = Path(__file__).parent / "new_presets.jsonl"
    presets = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q = row["query"]
        q = q[0] if isinstance(q, list) else q
        gold = row.get("gold", [])
        gold = [gold] if isinstance(gold, str) else list(gold)
        results = {}
        for mk, mv in (row.get("models") or {}).items():
            results[_MODEL_KEY_MAP.get(mk.lower(), mk)] = {
                "answer": mv.get("comp_answer", ""),  # compressed-route answer
                "full_answer": mv.get("full_answer", ""),
                "clf_prob": mv.get("clf_prob", 0.5),
                "route": mv.get("route"),
                "tokens_full": mv.get("tokens_full"),
                "tokens_compressed": mv.get("tokens_compressed"),
            }
        presets[str(row.get("id", q))] = {
            "context": row["context"],
            "query": q,
            "gold": gold,
            "results": results,
        }
    return presets


PRESETS = _load_presets()

_PRESET_KEYS = list(PRESETS.keys())
_PRESET_QUESTIONS = [PRESETS[_k]["query"] for _k in _PRESET_KEYS]
_QUERY_GOLD = {  # normalized query text → its saved gold answers (list)
    PRESETS[_k]["query"].strip().lower(): PRESETS[_k]["gold"] for _k in _PRESET_KEYS
}


_QUERY_PRESET = {  # normalized query text → its full preset (for canned answers)
    PRESETS[_k]["query"].strip().lower(): PRESETS[_k] for _k in _PRESET_KEYS
}


def _gold_for_query(query):
    """Saved gold answers (list) if this query matches a preset question, else None."""
    return _QUERY_GOLD.get((query or "").strip().lower())


def _active_preset_obj(query, context):
    """Preset whose query AND context match the current inputs → its canned answers;
    None for edited/custom inputs, so the model generates free-form instead."""
    p = _QUERY_PRESET.get((query or "").strip().lower())
    if p and (context or "").strip() == p["context"].strip():
        return p
    return None

# ── Handle ticker/selectbox question clicks ──
def _load_question(idx):
    """Queue a preset to load. The values are applied at the top of the next run
    (via _apply_pending_load), BEFORE the query/context widgets are created — the
    only safe way to set widget state, including from inside the plaque fragment.
    Mutating those widget keys directly here (mid-fragment) makes the fields vanish."""
    st.session_state["_pending_load"] = int(idx)


def _apply_pending_load():
    """Apply a queued preset load to the widget keys, before those widgets render."""
    idx = st.session_state.pop("_pending_load", None)
    if idx is None:
        return
    p = PRESETS[_PRESET_KEYS[idx % len(_PRESET_KEYS)]]
    st.session_state["query_input"] = p["query"]
    st.session_state["context_area"] = p["context"]
    st.session_state.pop("_search_url", None)
    st.session_state.pop("_sources", None)


# ── Multi-source search (avatar toggles) ─────────────────────────────
_SOURCE_PALETTE = list(COLORS.values())  # same accent triad used across the app
# Descriptive UA — Wikimedia (and others) throttle generic "Mozilla/5.0" bots.
_FETCH_UA = "OverflowGuardDemo/1.0 (Streamlit RAG demo) python-requests"
_KNOWN_HOSTS = {
    "en.wikipedia.org": "Wikipedia",
    "wikipedia.org": "Wikipedia",
    "geeksforgeeks.org": "GeeksforGeeks",
    "towardsdatascience.com": "Towards DS",
    "medium.com": "Medium",
    "stackoverflow.com": "Stack Overflow",
    "github.com": "GitHub",
    "arxiv.org": "arXiv",
}


def _source_name(url):
    """Human-friendly source name derived from the URL host."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in _KNOWN_HOSTS:
        return _KNOWN_HOSTS[host]
    labels = host.split(":")[0].split(".")
    core = labels[-2] if len(labels) >= 2 else labels[0]
    return core.capitalize()


def _fetch_page_text(url, max_chars):
    resp = requests.get(url, headers={"User-Agent": _FETCH_UA}, timeout=10)
    resp.raise_for_status()  # a 429/403 error page must be skipped, not scraped
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:max_chars]


def _host_of(url):
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _reg_domain(url):
    """Registrable domain (last two labels) — dedup key so subdomains collapse."""
    labels = _host_of(url).split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else (labels[0] if labels else "")


def search_sources(query, max_results=8, max_chars=4000, want=3):
    """Fetch up to `want` distinct-domain web sources for the query."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    sources = []
    seen_hosts = set()
    for r in results or []:
        if len(sources) >= want:
            break
        url = r.get("href")
        if not url:
            continue
        dom = _reg_domain(url)
        if dom in seen_hosts:  # one source per domain — avoids hammering one host
            continue
        try:
            text = _fetch_page_text(url, max_chars)
        except Exception:
            continue
        if not text:
            continue
        seen_hosts.add(dom)
        name = _source_name(url)
        sources.append(
            {
                "url": url,
                "name": name,
                "letter": name[0].upper(),
                "color": _SOURCE_PALETTE[len(sources) % len(_SOURCE_PALETTE)],
                "text": text,
                "active": len(sources) == 0,  # only the first source on by default
            }
        )
    return sources


def _context_from_sources(sources):
    """Concatenate the text of the currently active sources."""
    return "\n\n".join(s["text"] for s in sources if s["active"])


@st.cache_resource
def load_model(model_name):
    """Load the real OverflowRouter for this model; fall back to MockModel.

    Controlled by USE_REAL_MODELS (env OG_USE_REAL). Default is the mock; the
    real routers need a GPU and multi-GB weights, so we only attempt them when
    explicitly enabled. Also falls back if the framework can't be imported.
    """
    cls = ROUTER_CLASSES.get(model_name)
    if cls is None or not USE_REAL_MODELS:
        return MockModel(model_name)
    try:
        router = cls.from_pretrained(MODEL_PATH[model_name])
        router.model_name = model_name
        return router
    except Exception as e:
        st.warning(f"Failed to load {model_name}: {e}. Using MockModel.")
        return MockModel(model_name)

class _GpuSlot:
    """Which model is on the GPU, plus the lock that serializes GPU use.

    Must be process-wide: models come from st.cache_resource and are shared by
    every session, so per-session state can't know what another session put on
    the GPU. It can't be a plain module global either — Streamlit re-executes
    this script on every rerun, which would create a fresh lock each time.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.current = None  # name of the model currently parked on the GPU


@st.cache_resource
def _gpu_slot():
    return _GpuSlot()  # one per process, shared by all sessions


@contextmanager
def use_model(model_name):
    """Yield the model with exclusive use of the GPU; swap models if needed.

    Other sessions wait (with a notice) while one is generating. Mock-only runs
    touch no GPU, so they skip the lock and never queue behind each other.
    """
    if not USE_REAL_MODELS:
        yield load_model(model_name)
        return

    slot = _gpu_slot()
    if not slot.lock.acquire(blocking=False):
        notice = st.empty()
        notice.info("Waiting for the GPU — another visitor is running a model…")
        slot.lock.acquire()
        notice.empty()
    try:
        if slot.current != model_name:
            if slot.current is not None:
                load_model(slot.current).unpark_gpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            # recorded before parking: if parking fails halfway, the next swap
            # still offloads whatever part of this model reached the GPU
            slot.current = model_name
            load_model(model_name).park_gpu()
        yield load_model(model_name)
    finally:
        slot.lock.release()


# ── UI helpers ───────────────────────────────────────────────────────


def model_selector(label, key, default_index=1):
    if key not in st.session_state:
        st.session_state[key] = MODELS[default_index]
    st.caption(label)
    cols = st.columns(len(MODELS))
    for i, name in enumerate(MODELS):
        with cols[i]:
            if st.button(
                name,
                key=f"{key}_{name}",
                use_container_width=True,
                type="primary" if st.session_state[key] == name else "secondary",
            ):
                st.session_state[key] = name
                st.rerun()
    return st.session_state[key]


def mode_selector(label, key):
    if key not in st.session_state:
        st.session_state[key] = "Single"
    st.caption(label)
    cols = st.columns(3)
    for i, name in enumerate(["Single", "Compare", "Batch"]):
        with cols[i]:
            if st.button(
                name,
                key=f"{key}_{name}",
                use_container_width=True,
                type="primary" if st.session_state[key] == name else "secondary",
            ):
                st.session_state[key] = name
                st.rerun()
    return st.session_state[key]


def _stage_box(content, active=False, color="#333", exec_time=None):
    border = f"border:2px solid {color};" if active else "border:1px solid var(--t-border1);"
    badge = (
        f'<span style="position:absolute;right:12px;top:50%;transform:translateY(-50%);'
        f"background:var(--t-bg3);color:var(--t-text2);font-size:12px;padding:2px 10px;border-radius:6px;"
        f'border:1px solid var(--t-border2);">⏱ {exec_time:.3f}s</span>'
        if exec_time is not None
        else ""
    )
    return f'<div class="stage-box" style="{border}position:relative;">{content}{badge}</div>'


def _router_choice_html(color, clf_prob=None, route_mode=None):
    """CLF badge, then USING: [Memory tokens] / [Full context] — chosen one lit in accent."""
    clf_badge = ""
    if clf_prob is not None:
        clf_var = "badge_green" if route_mode == "compressed" else "badge_red"
        clf_badge = (
            f'<span style="background:var(--t-{clf_var}_bg);color:var(--t-{clf_var});'
            f"font-size:11px;font-weight:600;padding:2px 10px;border-radius:6px;\">"
            f"CLF {clf_prob:.2f}</span>"
        )
    chosen_mem = route_mode == "compressed"

    def box(label, active):
        if active:
            return (
                f'<span style="background:{color};color:#141414;padding:5px 14px;'
                f"border-radius:6px;font-size:12px;font-weight:700;white-space:nowrap;\">"
                f"✓&nbsp;{label}</span>"
            )
        return (
            f'<span style="background:{color}22;color:var(--t-text3);padding:5px 14px;'
            f"border-radius:6px;font-size:12px;font-weight:600;white-space:nowrap;"
            f'border:1px solid {color}33;">{label}</span>'
        )

    pipe = '<span style="color:var(--t-border1);font-size:18px;">|</span>'
    sep = '<span style="color:var(--t-text3);font-size:24px;">/</span>'
    using = '<span style="color:var(--t-text2);font-size:11px;font-weight:bold;">USING:</span>'
    parts = []
    if clf_badge:
        parts += [clf_badge, pipe]
    parts += [using, box("Memory tokens", chosen_mem), sep, box("Full context", not chosen_mem)]
    return (
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:16px;">'
        f'{"".join(parts)}</div>'
    )


def _header_html(model_name, color):
    return (
        f'<div style="background: linear-gradient(135deg, {color}18 0%, {color}08 100%);'
        f" border: 1px solid {color}30; border-radius: 10px;"
        f' padding: 16px 18px; margin: 12px 0;">'
        f'<span style="color:{color};font-size:20px;">●</span> '
        f'<span style="font-size:20px;font-weight:800;color:var(--t-text1);">{model_name}</span>'
        f"</div>"
    )


def _compression_bars_html(n_full, n_mem, color):
    """Full-context bar → (accent) compressed bar, length scaled by the mem/full ratio."""
    n_full = max(1, int(n_full))
    n_mem = max(0, int(n_mem))
    full_px = 560  # constant full-context bar
    mem_px = max(8, round(full_px * n_mem / n_full))
    return (
        f'<div style="color:{color};font-size:12px;margin-bottom:12px;">'
        f"● CONTEXT COMPRESSION</div>"
        f'<div style="display:flex;align-items:center;gap:18px;overflow-x:auto;">'
        f'<div style="flex:none;">'
        f'<div style="width:{full_px}px;height:16px;background:var(--t-split_full_bg);'
        f'border-radius:4px;"></div>'
        f'<div style="color:var(--t-text2);font-size:11px;margin-top:6px;">'
        f"full context · {n_full} tok</div></div>"
        f'<div style="color:var(--t-text3);font-size:30px;font-weight:800;'
        f'flex:none;line-height:0;">→</div>'
        f'<div style="flex:none;">'
        f'<div style="width:{mem_px}px;height:16px;background:{color};'
        f'border-radius:4px;"></div>'
        f'<div style="color:{color};font-size:11px;margin-top:6px;">'
        f"compressed · {n_mem} mem</div></div>"
        f"</div>"
    )


def _decoder_html(model_name, color):
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f"<span>"
        f'<span style="color:{color};">●</span> '
        f'<b style="color:var(--t-text1);margin-left:4px;">DECODER</b></span>'
        f'<span style="color:var(--t-text2);font-size:12px;">'
        f"frozen backbone{' · LoRA adapter' if model_name != 'xRAG' else ''}</span>"
        f"</div>"
    )


# ── CSS ──────────────────────────────────────────────────────────────
# Each viewer picks light/dark in ⋮ → Settings (palettes in .streamlit/config.toml).
# Both palettes below are always emitted; the theme script further down tags
# <html data-og-theme="light|dark"> from the background actually on screen, so
# the --t-* variables follow the viewer's choice instantly and per browser.


def _theme_vars():
    return {
        "dark": {
            "text1": "#E7E9EC", "text2": "#888", "text3": "#555",
            "bg1": "#0A0C0F", "bg2": "#12151A", "bg3": "#1a1d22",
            "border1": "#333", "border2": "#2a2d32",
            "badge_green": "#66ff7f", "badge_red": "#ff5252", "badge_yellow": "#EAC54F",
            "badge_green_bg": "#66ff7f22", "badge_red_bg": "#ff525222",
            "badge_yellow_bg": "#EAC54F22",
            "svg_bg": "#12151a", "svg_grid": "#1e2128",
            "answer_text": "#E7E9EC",
            "metric_val": "#eee", "metric_label": "#777", "metric_sub": "#888",
            "card_bg": "#12151a", "card_border": "transparent",
            "progress_bg": "#0a0c0f", "progress_outer": "#1a1d22",
            "split_full_bg": "#3a352e",
            "dot_muted": "#888", "dot_full": "#ccc", "accent_green": "#66ff7f",
        },
        "light": {
            "text1": "#212529", "text2": "#555", "text3": "#999",
            "bg1": "#D3D3D3", "bg2": "#F1F3F5", "bg3": "#E8EAED",
            "border1": "#bbb", "border2": "#ccc",
            "badge_green": "#1a7a2e", "badge_red": "#b22222", "badge_yellow": "#8a6a00",
            "badge_green_bg": "#1a7a2e22", "badge_red_bg": "#b2222222",
            "badge_yellow_bg": "#8a6a0022",
            "svg_bg": "#F1F3F5", "svg_grid": "#ddd",
            "answer_text": "#212529",
            "metric_val": "#212529", "metric_label": "#666", "metric_sub": "#888",
            "card_bg": "#F1F3F5", "card_border": "#ddd",
            "progress_bg": "#ddd", "progress_outer": "#E8EAED",
            "split_full_bg": "#ccc",
            "dot_muted": "#777", "dot_full": "#444", "accent_green": "#2ea84a",
        },
    }


def get_theme_css():
    def props(vals):
        return " ".join(f"--t-{k}: {v};" for k, v in vals.items())

    tv = _theme_vars()
    # dark is the fallback until the script has run
    return f"""<style>
:root {{ {props(tv["dark"])} }}
:root[data-og-theme="light"] {{ {props(tv["light"])} }}
.stage-box {{ background: var(--t-bg2); border-radius: 8px; padding: 12px; margin: 6px 0; }}
.mem-scroll {{ overflow-x: auto; white-space: nowrap; padding: 8px 0; }}
.mem-vec {{
    display: inline-block; background: var(--t-bg1); padding: 6px 8px; border-radius: 4px;
    min-width: 52px; text-align: center; margin-right: 6px; vertical-align: top;
    color: var(--t-text1);
}}
section[data-testid="stSidebar"] button {{ padding: 4px 8px; font-size: 13px; }}
section[data-testid="stSidebar"] button[kind="primary"]:active {{ color: #000 !important; }}
</style>"""


st.markdown(get_theme_css(), unsafe_allow_html=True)

components.html(
    """
<script>
const C = {"xRAG": "#5ED1FF", "PISCO": "#8238D9", "OSCAR": "#FFAF5E"};
const doc = parent.document;

// Tag <html> with the theme actually on screen (Streamlit's own light/dark,
// chosen per viewer in Settings), judged by the app background's luminance.
function syncTheme() {
    const app = doc.querySelector('.stApp');
    if (!app) return;
    const m = getComputedStyle(app).backgroundColor.match(/\\d+(\\.\\d+)?/g);
    if (!m) return;
    const [r, g, b] = m.map(Number);
    const theme = (0.299 * r + 0.587 * g + 0.114 * b) > 128 ? "light" : "dark";
    if (doc.documentElement.dataset.ogTheme !== theme) {
        doc.documentElement.dataset.ogTheme = theme;
    }
}

function paint() {
    parent.document.querySelectorAll('section[data-testid="stSidebar"] button[kind="primary"]').forEach(b => {
        const c = C[b.textContent.trim()];
        if (c) {
            b.style.backgroundColor = c;
            b.style.borderColor = c;
            b.style.color = "#000000";      // <-- black text for active buttons
        } else {
            b.style.backgroundColor = "#ffffff";
            b.style.borderColor = "#ffffff";
            b.style.color = "#000000";
        }
    });
}

function tick() { syncTheme(); paint(); }

tick();
new MutationObserver(tick).observe(doc.body, {
    childList: true,
    subtree: true
});
// a theme switch swaps Streamlit's stylesheet in <head> without touching <body>
new MutationObserver(syncTheme).observe(doc.head, { childList: true, subtree: true });
</script>
""",
    height=0,
)
# Keep the QUESTION/CONTEXT text alive across reruns that abort before the
# input widgets render. A sidebar button (mode/model) calls st.rerun()
# before line ~800, so those widgets aren't instantiated that run — Streamlit
# would then garbage-collect their keyed state and blank the fields. Re-committing
# the values as plain session_state each run (the documented persistence trick)
# promotes them out of widget-state so they survive.
for _persist_key in ("query_input", "context_area"):
    if _persist_key in st.session_state:
        st.session_state[_persist_key] = st.session_state[_persist_key]

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    # Question picker (carousel + dropdown) now lives in the main area, Section 1.
    mode = mode_selector("MODE", "mode")
    st.markdown("---")

    model_a = model_selector("MODEL A", "model_a", default_index=1)
    _default_ta = load_model(model_a).routing_threshold
    if st.session_state.get("_prev_model_a") != model_a:
        st.session_state["ta"] = _default_ta
        st.session_state["_prev_model_a"] = model_a
    threshold_a = st.slider(
        "Router threshold",
        0.0,
        1.0,
        key="ta",
        step=0.01,
        help="0 = full context, 1 = compress every chunk",
    )
    la, _, ra = st.columns(3)
    la.caption("Full context")
    ra.caption(
        '<div style="text-align:right;">Always compress</div>', unsafe_allow_html=True
    )

    if mode == "Compare":
        st.markdown("---")
        model_b = model_selector("MODEL B", "model_b", default_index=2)
        _default_tb = load_model(model_b).routing_threshold
        if st.session_state.get("_prev_model_b") != model_b or "tb" not in st.session_state:
            st.session_state["tb"] = _default_tb
            st.session_state["_prev_model_b"] = model_b
        threshold_b = st.slider(
            "Router threshold", 0.0, 1.0, key="tb", step=0.01
        )
        lb, _, rb = st.columns(3)
        lb.caption("Full context")
        rb.caption(
            '<div style="text-align:right;">Always compress</div>',
            unsafe_allow_html=True,
        )
    else:
        model_b = model_a
        threshold_b = threshold_a

def _render_source_pills():
    """Web-source pills: click one to toggle it into/out of the CONTEXT."""
    sources = st.session_state.get("_sources")
    if not sources:  # nothing until a web search has actually pulled sources
        return
    st.markdown(
        '<div style="font-size:11px;font-weight:700;letter-spacing:1px;color:var(--t-text3);'
        'text-transform:uppercase;margin:16px 0 8px;">Web sources</div>',
        unsafe_allow_html=True,
    )

    css = [
        "<style>",
        # lay the pills out in a horizontal wrapping row
        ".st-key-srcpills{flex-direction:row!important;flex-wrap:wrap!important;"
        "gap:10px!important;align-items:center!important;}",
        ".st-key-srcpills [data-testid='stElementContainer']{width:auto!important;}",
        ".st-key-srcpills button{width:auto!important;white-space:nowrap;"
        "border-radius:999px!important;padding:7px 18px!important;font-size:13px!important;}",
    ]
    for i, s in enumerate(sources):
        c = s["color"]
        css.append(
            f'.st-key-srcpill_{i} button[kind="primary"]{{background:{c}1f!important;'
            f"border:1px solid {c}!important;color:{c}!important;font-weight:700;}}"
            f'.st-key-srcpill_{i} button[kind="secondary"]{{background:transparent!important;'
            f"border:1px solid var(--t-border1)!important;color:var(--t-text2)!important;}}"
        )
    css.append("</style>")
    st.markdown("".join(css), unsafe_allow_html=True)

    with st.container(key="srcpills"):
        for i, s in enumerate(sources):
            label = ("✓ " if s["active"] else "+ ") + s["name"]
            if st.button(
                label, key=f"srcpill_{i}",
                type="primary" if s["active"] else "secondary",
                help=s["url"],
            ):
                s["active"] = not s["active"]
                # Rebuild happens next run, before the context_area widget exists.
                st.session_state["_rebuild_ctx"] = True
                st.rerun()


def _section_header(title, subtitle):
    """Plain bold step label with a muted subtitle."""
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:12px;margin:18px 0 8px;flex-wrap:wrap;">'
        f'<span style="font-size:14px;font-weight:800;letter-spacing:1.5px;'
        f'color:var(--t-text1);text-transform:uppercase;">{title}</span>'
        f'<span style="font-size:11px;color:var(--t-text3);">{subtitle}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


@st.fragment(run_every=1.0)
def _featured_plaque():
    """Auto-rotating preset: question text + Try this button, inside one colored box."""
    n = len(_PRESET_QUESTIONS)
    st.session_state.setdefault("_feat_idx", 0)
    st.session_state.setdefault("_feat_last", time.time())
    now = time.time()
    if now - st.session_state["_feat_last"] >= 3.0:  # auto-advance every 3s
        st.session_state["_feat_idx"] = (st.session_state["_feat_idx"] + 1) % n
        st.session_state["_feat_last"] = now
    idx = st.session_state["_feat_idx"]
    accent = list(COLORS.values())[idx % len(COLORS)]
    # Tint the whole container with the current accent (re-emitted each tick to rotate).
    st.markdown(
        f"<style>.st-key-plaque{{"
        f"background:linear-gradient(135deg,{accent}22 0%,{accent}06 100%);"
        f"border:1px solid {accent}45;border-radius:10px;"
        f"padding:6px 10px 6px 18px;margin-bottom:10px;}}"
        # Try this button tinted with the same accent as the plaque
        f".st-key-try_this button{{background:{accent}2e!important;"
        f"border:1px solid {accent}!important;color:{accent}!important;"
        f"white-space:nowrap;font-size:14px;}}"
        f".st-key-try_this button:hover{{background:{accent}47!important;"
        f"color:{accent}!important;border-color:{accent}!important;}}"
        f"</style>",
        unsafe_allow_html=True,
    )
    with st.container(key="plaque"):
        # No vertical_alignment: both columns top-align, and the 40px question div
        # (matching the button height) self-centers its text on the same line.
        q_col, b_col = st.columns([6, 1])
        with q_col:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;overflow:hidden;'
                f'white-space:nowrap;height:40px;">'
                f'<span style="color:{accent};font-size:12px;font-weight:bold;'
                f'letter-spacing:1px;flex:none;">● QUESTION #{idx + 1:02d}</span>'
                f'<span style="color:var(--t-text1);font-size:16px;font-weight:500;'
                f'overflow:hidden;text-overflow:ellipsis;">{_PRESET_QUESTIONS[idx]}</span></div>',
                unsafe_allow_html=True,
            )
        with b_col:
            if st.button("Try this ▸", key="try_this", use_container_width=True):
                _load_question(idx)
                st.rerun()  # full rerun so the QUESTION/CONTEXT fields pick up the values


def _question_dropdown():
    """Browse-all picker for any of the questions."""
    q_options = [f"#{i+1} {q}" for i, q in enumerate(_PRESET_QUESTIONS)]
    picked = st.selectbox(
        "select_question", q_options, key="_qpick",
        index=None, placeholder="Browse all questions",
        label_visibility="collapsed",
    )
    if picked is not None and picked != st.session_state.get("_last_picked"):
        st.session_state["_last_picked"] = picked
        _load_question(q_options.index(picked))
        st.rerun()


if mode != "Batch":
    _apply_pending_load()  # apply a queued "Try this"/Browse selection before widgets
    preset = PRESETS[_PRESET_KEYS[0]]  # initial question/context

    # Bigger fonts in the input fields; keep the pill/action buttons on one line.
    st.markdown(
        "<style>"
        ".st-key-query_input input{font-size:18px!important;padding:15px 16px!important;}"
        ".st-key-context_area textarea{font-size:16px!important;line-height:1.6!important;}"
        ".st-key-try_this button{white-space:nowrap;font-size:14px;}"
        ".st-key-_qpick [data-baseweb='select']{font-size:14px;}"
        ".st-key-run_inf button{white-space:nowrap;font-size:16px;}"
        ".st-key-ctx_search button{white-space:nowrap;font-size:13px;}"
        # scaled search glyph appended after the label (bigger than the text)
        ".st-key-ctx_search button::after{content:'⌕';font-size:19px;font-weight:400;"
        "line-height:0;margin-left:6px;vertical-align:-3px;}"
        "</style>",
        unsafe_allow_html=True,
    )

    # ── PICK A PRESET: question + Try this inside the colored plaque
    _section_header("Pick a question", "swaps every few seconds")
    _featured_plaque()

    # Browse all questions │ slash divider │ type-your-own input — one row.
    # Center-aligned so Browse rides higher, level with the input group.
    _browse_col, _div_col, _own_col = st.columns(
        [3, 0.4, 7], vertical_alignment="center"
    )
    with _browse_col:
        _question_dropdown()
    with _div_col:
        st.markdown(
            '<div style="height:60px;display:flex;align-items:center;'
            'justify-content:center;"><div style="width:2px;height:50px;'
            'background:var(--t-text2);opacity:0.7;transform:rotate(18deg);'
            'border-radius:2px;"></div></div>',
            unsafe_allow_html=True,
        )
    with _own_col:
        st.markdown(
            '<div style="font-size:12px;color:var(--t-text3);'
            'margin:10px 0 2px;">&#9998; Type your own</div>',
            unsafe_allow_html=True,
        )
        if "query_input" not in st.session_state:
            st.session_state["query_input"] = preset["query"]
        query = st.text_input(
            "QUESTION", key="query_input", label_visibility="collapsed",
            placeholder="Ask anything…",
        )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)  # gap to CONTEXT

    # ── CONTEXT
    h_col, s_col = st.columns([2, 1], vertical_alignment="center")
    with h_col:
        _section_header(
            "Context", "optional — paste your own, or pull it from the web"
        )
    with s_col:
        search_btn = st.button(
            "Search web for context", key="ctx_search", use_container_width=True
        )

    if search_btn and query.strip():
        with st.spinner("Searching the web for context…"):
            sources = search_sources(query)
        if sources:
            st.session_state["_sources"] = sources
            # Defer the context rebuild to the next run (before the widget exists),
            # same reliable path the pills use — a direct set here doesn't override
            # the already-instantiated text_area across the rerun.
            st.session_state["_rebuild_ctx"] = True
            st.rerun()
        else:
            st.warning("No results found.")

    if st.session_state.pop("_rebuild_ctx", False):
        _srcs = st.session_state.get("_sources")
        if _srcs is not None:
            st.session_state["context_area"] = _context_from_sources(_srcs)
    if "context_area" not in st.session_state:
        st.session_state["context_area"] = preset["context"]
    context = st.text_area(
        "CONTEXT", key="context_area", height=170,
        label_visibility="collapsed", placeholder="Context the model will read…",
    )

    # ── WEB SOURCES (pills)
    _render_source_pills()

    # ── Run inference (bottom-right)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    _ri_sp, _ri_col = st.columns([3, 1])
    with _ri_col:
        run_inference_btn = st.button(
            "Run inference ▸", key="run_inf", use_container_width=True
        )

    st.markdown("---")


ARROW = '<div style="text-align:center;color:#555;margin:2px 0;">↓</div>'

# ── Pipeline rendering (uses generator) ──────────────────────────────


def _setup_column(model, container):
    """Create placeholders and return (phs dict, state dict)."""
    color = COLORS[model.model_name]
    with container:
        phs = {
            k: st.empty()
            for k in (
                "header",
                "tokens",
                "arrow1",
                "router",
                "arrow2",
                "decoder",
                "answer_hdr",
                "answer",
                "gold",
                "metrics",
            )
        }
    phs["header"].markdown(
        _header_html(model.model_name, color),
        unsafe_allow_html=True,
    )
    phs["answer_hdr"].markdown(
        '<div style="margin-top:8px;">'
        '<span style="color:var(--t-text2);font-size:12px;font-weight:bold;">ANSWER</span></div>',
        unsafe_allow_html=True,
    )
    state = {
        "model_name": model.model_name,
        "displayed": "",
        "n_chunks": 0,
        "n_comp": 0,
        "n_full": 0,
        "n_mem": 0,
        "comp_html": "",
        "result": None,
    }
    return phs, state


def _handle_stage(update, phs, state, color):
    """Process one generator yield and update the corresponding placeholder."""
    stage = update["stage"]
    et = update.get("exec_time")
    if stage == "compression":
        embs = update["compressed_embs"]
        state["compression_time"] = et
        state["n_chunks"] = embs.shape[0]
        state["n_mem"] = embs.shape[0] * embs.shape[1]
        state["n_full"] = update.get("tokens_full", state["n_mem"])
        state["comp_html"] = _compression_bars_html(
            state["n_full"], state["n_mem"], color
        )
        phs["tokens"].markdown(
            _stage_box(state["comp_html"], active=True, color=color, exec_time=et),
            unsafe_allow_html=True,
        )
        phs["arrow1"].markdown(ARROW, unsafe_allow_html=True)
    elif stage == "router":
        state["n_comp"] = state["n_chunks"] if update["mode"] == "compressed" else 0
        state["router_time"] = et
        state["clf_prob"] = update.get("clf_prob")
        state["route_mode"] = update["mode"]
        inner = _router_choice_html(
            color,
            clf_prob=state["clf_prob"],
            route_mode=update["mode"],
        )
        phs["router"].markdown(
            _stage_box(inner, active=True, color=color, exec_time=et),
            unsafe_allow_html=True,
        )
        phs["arrow2"].markdown(ARROW, unsafe_allow_html=True)
    elif stage == "generation":
        phs["decoder"].markdown(
            _stage_box(_decoder_html(state["model_name"], color), active=True, color=color),
            unsafe_allow_html=True,
        )
    elif stage == "token":
        state["displayed"] += update["token"]
        phs["answer"].markdown(
            f'<div style="font-size:15px;min-height:60px;color:var(--t-answer_text);">{state["displayed"]}▌</div>',
            unsafe_allow_html=True,
        )
    elif stage == "done":
        state["result"] = update["result"]


def _finalize(phs, state, model, query):
    """Deactivate all boxes and render final state."""
    color = COLORS[model.model_name]
    result = state["result"]
    # Rebuild the bars from the final result — real routers only report
    # tokens_full at the "done" stage, not during compression.
    n_full = result.get("tokens_full") or state["n_full"]
    n_mem = result.get("tokens_compressed") or state["n_mem"]
    comp_html = _compression_bars_html(n_full, n_mem, color)

    phs["header"].markdown(
        _header_html(model.model_name, color),
        unsafe_allow_html=True,
    )
    phs["tokens"].markdown(
        _stage_box(comp_html, exec_time=state["compression_time"]),
        unsafe_allow_html=True,
    )
    phs["router"].markdown(
        _stage_box(
            _router_choice_html(
                color,
                clf_prob=state.get("clf_prob"),
                route_mode=state.get("route_mode"),
            ),
            exec_time=state["router_time"],
        ),
        unsafe_allow_html=True,
    )
    phs["decoder"].markdown(_stage_box(_decoder_html(model.model_name, color)), unsafe_allow_html=True)
    # Known query → substring-match generated answer against its saved gold;
    # unknown (custom) query → neutral yellow "custom query" badge.
    gold = _gold_for_query(query)  # list of accepted gold answers, or None
    if not gold:
        badge = (
            '<span style="background:var(--t-badge_yellow_bg);color:var(--t-badge_yellow);'
            'font-size:11px;font-weight:600;padding:2px 12px;border-radius:6px;">'
            "CUSTOM QUERY</span>"
        )
    else:
        pred = (result["prediction"] or "").lower()
        correct = any(g.strip().lower() in pred for g in gold)
        pill_var = "badge_green" if correct else "badge_red"
        mark = "✓" if correct else "✗"
        badge = (
            f'<span style="background:var(--t-{pill_var}_bg);color:var(--t-{pill_var});'
            f'font-size:11px;font-weight:600;padding:2px 12px;border-radius:6px;">'
            f"{mark}&nbsp;&nbsp;SUBSTRING MATCH</span>"
        )
    phs["answer_hdr"].markdown(
        f'<div style="margin-top:8px;margin-bottom:8px;display:flex;'
        f'justify-content:space-between;align-items:center;">'
        f'<span style="color:var(--t-text2);font-size:12px;font-weight:bold;">ANSWER</span>'
        f"{badge}</div>",
        unsafe_allow_html=True,
    )
    phs["answer"].markdown(
        f'<div style="font-size:15px;min-height:60px;color:var(--t-answer_text);">{result["prediction"]}</div>',
        unsafe_allow_html=True,
    )
    if gold:
        phs["gold"].caption(f"gold · {', '.join(gold)}")
    tok_full = result["tokens_full"] or 1
    pct_saved = round(result["tokens_saved"] / tok_full * 100)
    phs["metrics"].markdown(
        f'<div style="display:flex;gap:24px;margin-top:8px;padding-top:8px;border-top:1px solid var(--t-border1);">'
        f'<div><div style="color:var(--t-metric_label);font-size:10px;">% SAVED</div>'
        f'<div style="font-size:20px;font-weight:bold;color:{color};">{pct_saved}%</div></div>'
        f'<div><div style="color:var(--t-metric_label);font-size:10px;">CONTEXT TOKENS</div>'
        f'<div style="font-size:20px;font-weight:bold;color:var(--t-metric_val);">{result["tokens_compressed"]}</div>'
        f'<div style="color:#E0703F;font-size:11px;">saved {result["tokens_saved"]}</div></div>'
        f'<div><div style="color:var(--t-metric_label);font-size:10px;">TOKENS FULL</div>'
        f'<div style="font-size:20px;font-weight:bold;color:var(--t-metric_val);">{result["tokens_full"]}</div></div>'
        f'<div><div style="color:var(--t-metric_label);font-size:10px;">EXECUTION TIME</div>'
        f'<div style="font-size:20px;font-weight:bold;color:var(--t-metric_val);">{round(result["exec_time"], 3)}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    return result


def _build_preset(preset_obj, model_name):
    """This model's saved answers from a preset (mock-only canned output), or None."""
    if not preset_obj:
        return None
    return preset_obj.get("results", {}).get(model_name)


def run_pipeline_live(model, context, query, threshold, container, preset_obj=None):
    """Drive a single model's pipeline into Streamlit placeholders."""
    color = COLORS[model.model_name]
    preset = _build_preset(preset_obj, model.model_name)
    phs, state = _setup_column(model, container)

    _kw = {"threshold": threshold}
    if isinstance(model, MockModel):  # preset (canned answers) is mock-only
        _kw["preset"] = preset
    for update in model.run_pipeline_with_progress(context, query, **_kw):
        _handle_stage(update, phs, state, color)

    return _finalize(phs, state, model, query)


def run_comparison_live(
    m_left,
    m_right,
    context,
    query,
    th_a,
    th_b,
    col_left,
    col_right,
    preset_obj=None,
):
    """Drive two models with interleaved token streaming."""
    phs_l, st_l = _setup_column(m_left, col_left)
    phs_r, st_r = _setup_column(m_right, col_right)
    color_l = COLORS[m_left.model_name]
    color_r = COLORS[m_right.model_name]
    preset_l = _build_preset(preset_obj, m_left.model_name)
    preset_r = _build_preset(preset_obj, m_right.model_name)

    _kw_l = {"threshold": th_a}
    _kw_r = {"threshold": th_b}
    if isinstance(m_left, MockModel):  # preset (canned answers) is mock-only
        _kw_l["preset"] = preset_l
    if isinstance(m_right, MockModel):
        _kw_r["preset"] = preset_r
    gen_l = m_left.run_pipeline_with_progress(context, query, **_kw_l)
    gen_r = m_right.run_pipeline_with_progress(context, query, **_kw_r)

    # Interleave all stages — alternate next() calls so both columns animate together
    done_l = done_r = False
    while not done_l or not done_r:
        if not done_l:
            try:
                u = next(gen_l)
                _handle_stage(u, phs_l, st_l, color_l)
                if u["stage"] == "done":
                    done_l = True
            except StopIteration:
                done_l = True
        if not done_r:
            try:
                u = next(gen_r)
                _handle_stage(u, phs_r, st_r, color_r)
                if u["stage"] == "done":
                    done_r = True
            except StopIteration:
                done_r = True

    return _finalize(phs_l, st_l, m_left, query), _finalize(phs_r, st_r, m_right, query)


SCALE_DATA = json.loads((Path(__file__).parent / "scale_data.json").read_text())


def _scale_metric_card(label, value, sub="", accent=None):
    if accent:
        def tint(pct):  # accent may be a hex color or a CSS var
            return f"color-mix(in srgb, {accent} {pct}%, transparent)"

        bg = f"background:linear-gradient(135deg, {tint(9)} 0%, {tint(3)} 100%);border:1px solid {tint(19)};"
        label_color = accent
        value_color = accent
        sub_color = tint(60)
    else:
        bg = "background:var(--t-card_bg);"
        label_color = "var(--t-metric_label)"
        value_color = "var(--t-metric_val)"
        sub_color = "var(--t-metric_sub)"
    sub_html = (
        f'<div style="color:{sub_color};font-size:11px;margin-top:2px;">{sub}</div>'
        if sub
        else ""
    )
    return (
        f'<div style="{bg}border-radius:8px;padding:16px 20px;flex:1;min-width:150px;">'
        f'<div style="color:{label_color};font-size:11px;font-weight:bold;text-transform:uppercase;">{label}</div>'
        f'<div style="color:{value_color};font-size:28px;font-weight:800;margin-top:4px;">{value}</div>'
        f"{sub_html}</div>"
    )


def _cost_quality_svg(comp_all, router_pt, full_ctx, color, w=700, h=320):
    """SVG scatter: x = avg tokens/query, y = EM accuracy. 3 dots + dashed line."""
    pad_l, pad_r, pad_t, pad_b = 55, 25, 25, 40
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    x_min, x_max = 0, 300
    y_min, y_max = 20, 105

    def sx(v):
        return pad_l + (v - x_min) / (x_max - x_min) * pw

    def sy(v):
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * ph
    # theme colors via CSS vars (style=, since var() isn't valid in SVG attributes)
    svg_bg = "var(--t-svg_bg)"
    svg_grid = "var(--t-svg_grid)"
    svg_text = "var(--t-text3)"

    svg = (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;display:block;">'
        f'<rect width="{w}" height="{h}" style="fill:{svg_bg}" rx="8"/>'
    )

    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        yp = sy(yv)
        svg += f'<line x1="{pad_l}" y1="{yp}" x2="{w - pad_r}" y2="{yp}" style="stroke:{svg_grid}" stroke-width="1"/>'
        svg += f'<text x="{pad_l - 8}" y="{yp + 4}" style="fill:{svg_text}" font-size="10" text-anchor="end" font-family="monospace">{yv:.0f}%</text>'

    for i in range(7):
        xv = i * 50
        xp = sx(xv)
        svg += f'<line x1="{xp}" y1="{pad_t}" x2="{xp}" y2="{h - pad_b}" style="stroke:{svg_grid}" stroke-width="1"/>'
        svg += f'<text x="{xp}" y="{h - pad_b + 16}" style="fill:{svg_text}" font-size="10" text-anchor="middle" font-family="monospace">{xv:.0f}</text>'

    svg += f'<text x="{w // 2}" y="{h - 4}" style="fill:{svg_text}" font-size="10" text-anchor="middle" font-family="monospace">avg tokens / query</text>'
    svg += f'<text x="12" y="{h // 2}" style="fill:{svg_text}" font-size="10" text-anchor="middle" font-family="monospace" transform="rotate(-90,12,{h // 2})">Accuracy</text>'

    dot_muted = "var(--t-dot_muted)"
    dot_full = "var(--t-dot_full)"
    points = [
        ("Compress-all", comp_all[0], comp_all[1], dot_muted),
        ("Router", router_pt[0], router_pt[1], color),
        ("Full context", full_ctx[0], full_ctx[1], dot_full),
    ]
    px = [sx(p[1]) for p in points]
    py = [sy(p[2]) for p in points]

    svg += f'<line x1="{px[0]}" y1="{py[0]}" x2="{px[1]}" y2="{py[1]}" stroke="{color}" stroke-width="1" stroke-dasharray="5,4" opacity="0.6"/>'
    svg += f'<line x1="{px[1]}" y1="{py[1]}" x2="{px[2]}" y2="{py[2]}" stroke="{color}" stroke-width="1" stroke-dasharray="5,4" opacity="0.6"/>'

    # (dx, dy for label, dy for sub, text-anchor)
    # Compress-all: top-right of dot; Router: bottom-right; Full context: top-right
    label_cfg = [(12, -20, -7, "start"), (12, 16, 29, "start"), (-12, -20, -7, "end")]
    for i, (lab, xv, yv, dc) in enumerate(points):
        cx, cy = px[i], py[i]
        r = 7 if i == 1 else 5
        svg += f'<circle cx="{cx}" cy="{cy}" r="{r}" style="fill:{dc}"/>'
        dx, dy1, dy2, anchor = label_cfg[i]
        svg += f'<text x="{cx + dx}" y="{cy + dy1}" style="fill:{dc}" font-size="11" text-anchor="{anchor}" font-family="monospace">{lab}</text>'
        svg += f'<text x="{cx + dx}" y="{cy + dy2}" style="fill:{svg_text}" font-size="9" text-anchor="{anchor}" font-family="monospace">{xv:.0f} tok · {yv:.1f}%</text>'

    svg += "</svg>"
    return svg


def _run_scale_page(model_name, threshold):
    color = COLORS[model_name]
    tpc = TOKENS_PER_CHUNK[model_name]
    samples = SCALE_DATA[model_name]
    n = len(samples)

    st.markdown(
        f'<div style="background: linear-gradient(135deg, {color}18 0%, {color}08 100%);'
        f" border: 1px solid {color}30; border-radius: 10px;"
        f" padding: 16px 18px; margin: 0 0 16px 0;"
        f' display:flex;align-items:center;justify-content:space-between;">'
        f"<div>"
        f'<span style="color:{color};font-size:20px;">●</span> '
        f'<span style="font-size:20px;font-weight:800;color:var(--t-text1);">{model_name}</span>'
        f'<span style="color:var(--t-text2);font-size:13px;margin-left:16px;">'
        f"{tpc} mem tokens/chunk · threshold {threshold:.2f} · {n} samples</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    run_btn = st.button("Run batch ▸", use_container_width=True)

    ph_progress = st.empty()
    ph_split = st.empty()
    ph_metrics = st.empty()
    ph_plot = st.empty()

    if not run_btn:
        ph_progress.markdown(
            f'<div style="background:var(--t-bg2);border-radius:8px;padding:12px;color:var(--t-text3);text-align:center;">'
            f'Press "Run batch" to evaluate {n} samples</div>',
            unsafe_allow_html=True,
        )
        return

    total_tokens_saved = 0
    total_tokens_full = 0
    n_compressed = 0
    n_full = 0
    router_correct = 0
    total_tokens_used = 0

    comp_all_correct = 0
    full_all_correct = 0
    comp_all_tokens = 0
    full_all_tokens = 0

    for i, sample in enumerate(samples):
        use_comp = sample["clf_prob"] <= threshold

        if use_comp:
            n_compressed += 1
            is_correct = sample["compressed_correct"]
            tokens_used = sample["tokens_compressed"]
        else:
            n_full += 1
            is_correct = sample["full_correct"]
            tokens_used = sample["tokens_full"]

        if is_correct:
            router_correct += 1

        total_tokens_full += sample["tokens_full"]
        total_tokens_used += tokens_used
        total_tokens_saved += sample["tokens_full"] - tokens_used

        if sample["compressed_correct"]:
            comp_all_correct += 1
        if sample["full_correct"]:
            full_all_correct += 1
        comp_all_tokens += sample["tokens_compressed"]
        full_all_tokens += sample["tokens_full"]

        done = i + 1
        # Throttle rendering to keep every placeholder on the SAME frame — updating
        # 4 separate st.empty()s 1000× desyncs them (bar lags the numbers).
        if done % 20 != 0 and done != n:
            continue
        pct = done / n

        ph_progress.markdown(
            f'<div style="background:var(--t-progress_outer);border-radius:8px;padding:8px 12px;margin:8px 0;">'
            f'<div style="color:var(--t-text2);font-size:11px;margin-bottom:4px;">{done}/{n} samples</div>'
            f'<div style="background:var(--t-progress_bg);border-radius:4px;height:12px;overflow:hidden;">'
            f'<div style="background:{color};height:100%;width:{pct * 100:.1f}%;'
            f'border-radius:4px;"></div></div></div>',
            unsafe_allow_html=True,
        )

        comp_pct = n_compressed / done * 100
        full_pct = n_full / done * 100
        ph_split.markdown(
            f'<div style="margin:8px 0;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
            f'<span style="color:{color};font-size:11px;font-weight:bold;">COMPRESSED {comp_pct:.0f}%</span>'
            f'<span style="color:var(--t-text2);font-size:11px;font-weight:bold;">FULL {full_pct:.0f}%</span></div>'
            f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;">'
            f'<div style="background:{color};width:{comp_pct}%;"></div>'
            f'<div style="background:var(--t-split_full_bg);width:{full_pct}%;"></div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )

        saved_pct = total_tokens_saved / max(total_tokens_full, 1) * 100
        router_acc = router_correct / done * 100
        avg_tok = total_tokens_used / done

        ph_metrics.markdown(
            '<div style="display:flex;gap:12px;margin:8px 0;">'
            + _scale_metric_card(
                "Tokens saved",
                f"{saved_pct:.1f}%",
                f"{total_tokens_saved:,} / {total_tokens_full:,}",
                accent="var(--t-accent_green)",
            )
            + _scale_metric_card(
                "Router accuracy", f"{router_acc:.1f}%", f"{router_correct} / {done}"
            )
            + _scale_metric_card(
                "Avg tokens/query",
                f"{avg_tok:.0f}",
                f"vs {total_tokens_full / done:.0f} full",
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        comp_all_acc = comp_all_correct / done * 100
        full_all_acc = full_all_correct / done * 100
        avg_comp_tok = comp_all_tokens / done
        avg_full_tok = full_all_tokens / done

        ph_plot.markdown(
            _cost_quality_svg(
                comp_all=(avg_comp_tok, comp_all_acc),
                router_pt=(avg_tok, router_acc),
                full_ctx=(avg_full_tok, full_all_acc),
                color=color,
            ),
            unsafe_allow_html=True,
        )

        time.sleep(0.06)  # per rendered frame (~50 frames) → smooth ~3s sweep


# ── Main layout ──────────────────────────────────────────────────────


if mode == "Batch":
    _, col_mid, _ = st.columns([1, 3, 1])
    with col_mid:
        _run_scale_page(model_a, threshold_a)

elif mode == "Compare":
    if run_inference_btn:
        active = _active_preset_obj(query, context)  # match by current query+context
        col_left, col_right = st.columns(2)
        m_left = MockModel(model_a)
        try:
            with use_model(model_b) as m_right:
                left_result, right_result = run_comparison_live(
                    m_left,
                    m_right,
                    context,
                    query,
                    threshold_a,
                    threshold_b,
                    col_left,
                    col_right,
                    preset_obj=active,
                )
        except Exception as e:
            st.error(f"GPU error: {e}. Falling back to MockModel.")
            m_right = MockModel(model_b)
            left_result, right_result = run_comparison_live(
                m_left,
                m_right,
                context,
                query,
                threshold_a,
                threshold_b,
                col_left,
                col_right,
                preset_obj=active,
            )

else:
    if run_inference_btn:
        active = _active_preset_obj(query, context)  # match by current query+context
        (col_center,) = st.columns([1])
        try:
            with use_model(model_a) as model:
                run_pipeline_live(
                    model,
                    context,
                    query,
                    threshold_a,
                    col_center,
                    preset_obj=active,
                )
        except Exception as e:
            st.error(f"GPU error: {e}. Falling back to MockModel.")
            model = MockModel(model_a)
            run_pipeline_live(
                model,
                context,
                query,
                threshold_a,
                col_center,
                preset_obj=active,
            )
