"""
PISCO Live Inference Demo — Streamlit app.

Run with:  streamlit run app.py
"""

import gc
import json
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoModel

from mock_model import MockModel

st.set_page_config(
    page_title="Inference Demo", layout="wide", initial_sidebar_state="expanded"
)

# ── Constants ────────────────────────────────────────────────────────
MODELS = ["xRAG", "PISCO", "OSCAR"]
COLORS = {"xRAG": "#5ED1FF", "PISCO": "#8238D9", "OSCAR": "#FFAF5E"}
COLORS_BACK_MAP = {v: k for k, v in COLORS.items()}
MODEL_PATH = {"PISCO": "/models/pisco-router", "OSCAR": "/models/oscar-mistral-7B", "xRAG": "Hannibal046/xrag-7b"}
TOKENS_PER_CHUNK = {"xRAG": 1, "PISCO": 8, "OSCAR": 8}

PRESETS = json.loads((Path(__file__).parent / "presets.json").read_text())
XRAG_CLF_DIR = "./xrag-clf"
_GPU_LOCK = threading.Lock()

@st.cache_resource
def load_model(model_name):
    path = MODEL_PATH.get(model_name)
    if not path:
        return MockModel(model_name)
    try:
        if model_name == "xRAG":
            from modelling_xrag_router import XRAGRouter
            m = XRAGRouter.from_pretrained(path)
            m.load_routing(XRAG_CLF_DIR)
            return m
        return AutoModel.from_pretrained(path, trust_remote_code=True).eval()
    except Exception as e:
        st.warning(f"Failed to load {model_name}: {e}. Using MockModel.")
        return MockModel(model_name)

def _ensure_gpu(model_name):
    """Move the selected model to GPU, offloading any previous one.

    Keeps the active model parked on GPU to reserve VRAM.
    MockModel is a no-op (no GPU needed).
    xRAG: parks LLM on GPU, SFR stays on CPU (swapped in only during compress).
    PISCO/OSCAR: moves the whole model to GPU.
    """
    current = st.session_state.get("_gpu_model")
    if current == model_name:
        return

    # Offload previous model
    if current and current != model_name:
        prev = load_model(current)
        if not isinstance(prev, MockModel):
            if hasattr(prev, "unpark_gpu"):
                prev.unpark_gpu()
            else:
                prev.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

    # Load new model to GPU
    model = load_model(model_name)
    if not isinstance(model, MockModel):
        if hasattr(model, "park_gpu"):
            model.park_gpu()
        else:
            model.to("cuda")

    st.session_state["_gpu_model"] = model_name


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
    for i, name in enumerate(["Single", "Comparison", "Scale"]):
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
        f"background:var(--t-bg3);color:var(--t-text2);font-size:11px;padding:2px 10px;border-radius:6px;"
        f'border:1px solid var(--t-border2);">⏱ {exec_time:.3f}s</span>'
        if exec_time is not None
        else ""
    )
    return f'<div class="stage-box" style="{border}position:relative;">{content}{badge}</div>'


def _router_chips_html(n_chunks, n_comp, color, tpc, clf_prob=None, route_mode=None):
    chips = ""
    if n_comp > 0:
        for i in range(n_comp):
            chips += (
                f'<span style="display:inline-block;background:{color};'
                f"color:white;padding:4px 14px;border-radius:6px;"
                f'margin:2px;font-size:12px;font-family:monospace;">'
                f"{tpc}-mem</span>"
            )
    else:
        chips = (
            '<span style="display:inline-block;background:var(--t-split_full_bg);'
            "color:var(--t-text2);padding:4px 14px;border-radius:6px;"
            'margin:2px;font-size:12px;font-family:monospace;">'
            "full</span>"
        )
    clf_badge = ""
    if clf_prob is not None:
        clf_var = "badge_green" if route_mode == "compressed" else "badge_red"
        clf_badge = (
            f'<span style="background:var(--t-{clf_var}_bg);color:var(--t-{clf_var});font-size:11px;font-weight:600;'
            f'padding:2px 10px;border-radius:6px;margin-left:10px;">'
            f"CLF {clf_prob:.2f}</span>"
        )
    return (
        f'<span style="color:var(--t-text2);font-size:11px;font-weight:bold;'
        f'margin-right:12px;">{clf_badge} | ROUTER VERDICT:</span>{chips}'
    )


def _header_html(model_name, color):
    return (
        f'<div style="background: linear-gradient(135deg, {color}18 0%, {color}08 100%);'
        f" border: 1px solid {color}30; border-radius: 10px;"
        f' padding: 16px 18px; margin: 12px 0;">'
        f'<span style="color:{color};font-size:20px;">●</span> '
        f'<span style="font-size:22px;font-weight:800;color:var(--t-text1);">{model_name}</span>'
        f"</div>"
    )


def _mem_tokens_html(memory_tokens, n_comp, color):
    total = len(memory_tokens)
    h = (
        f'<div style="color:{color};font-size:12px;margin-bottom:6px;">'
        f"● MEMORY TOKENS &emsp;"
        f'<span style="color:var(--t-text2);">{n_comp} chunks → {total} '
        f"memory vectors · d=4096</span></div>"
        f'<div class="mem-scroll">'
    )
    for i, vec in enumerate(memory_tokens):
        vals = "<br>".join(f"{v:.2f}" for v in vec[:4]) + "<br>..."
        h += (
            f'<div class="mem-vec">'
            f'<div style="color:var(--t-text2);font-size:10px;">m{i + 1}</div>'
            f'<div style="color:var(--t-text1);font-size:11px;'
            f'font-family:monospace;">{vals}</div></div>'
        )
    h += 2 * (
        f'<div class="mem-vec"  style="opacity: 0;">'
        f'<div style="color:var(--t-text2);font-size:10px;">m{i + 1}</div>'
        f'<div style="color:var(--t-text1);font-size:11px;'
        f'font-family:monospace;">{vals}</div></div>'
    )
    h += "</div>"
    return h


def _decoder_html(color):
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f"<span>"
        f'<span style="color:{color};">●</span> '
        f'<b style="color:var(--t-text1);margin-left:4px;">DECODER</b></span>'
        f'<span style="color:var(--t-text2);font-size:12px;">'
        f"frozen backbone{' · LoRA adapter' if COLORS_BACK_MAP[color] != 'xRAG' else ''}</span>"
        f"</div>"
    )


# ── CSS ──────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def get_theme_css():
    dark = st.session_state.theme == "dark"
    if dark:
        vals = {
            "text1": "#E7E9EC", "text2": "#888", "text3": "#555",
            "bg1": "#0A0C0F", "bg2": "#12151A", "bg3": "#1a1d22",
            "border1": "#333", "border2": "#2a2d32",
            "badge_green": "#66ff7f", "badge_red": "#ff5252",
            "badge_green_bg": "#66ff7f22", "badge_red_bg": "#ff525222",
            "svg_bg": "#12151a", "svg_grid": "#1e2128",
            "answer_text": "#E7E9EC",
            "metric_val": "#eee", "metric_label": "#777", "metric_sub": "#888",
            "card_bg": "#12151a", "card_border": "transparent",
            "progress_bg": "#0a0c0f", "progress_outer": "#1a1d22",
            "split_full_bg": "#3a352e",
        }
    else:
        vals = {
            "text1": "#212529", "text2": "#555", "text3": "#999",
            "bg1": "#D3D3D3", "bg2": "#F1F3F5", "bg3": "#E8EAED",
            "border1": "#bbb", "border2": "#ccc",
            "badge_green": "#1a7a2e", "badge_red": "#b22222",
            "badge_green_bg": "#1a7a2e22", "badge_red_bg": "#b2222222",
            "svg_bg": "#F1F3F5", "svg_grid": "#ddd",
            "answer_text": "#212529",
            "metric_val": "#212529", "metric_label": "#666", "metric_sub": "#888",
            "card_bg": "#F1F3F5", "card_border": "#ddd",
            "progress_bg": "#ddd", "progress_outer": "#E8EAED",
            "split_full_bg": "#ccc",
        }

    props = "\n".join(f"  --t-{k}: {v};" for k, v in vals.items())
    return f"""<style>
:root {{ {props} }}
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

paint();
new MutationObserver(paint).observe(parent.document.body, {
    childList: true,
    subtree: true
});
</script>
""",
    height=0,
)
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Start with your current dark theme

def toggle_theme():
    if st.session_state.theme == "dark":
        # Switch to LIGHT theme
        st._config.set_option("theme.base", "light")
        st._config.set_option("theme.primaryColor", "#EF7849")
        st._config.set_option("theme.backgroundColor", "#F8F9FA")
        st._config.set_option("theme.secondaryBackgroundColor", "#F1F3F5")
        st._config.set_option("theme.textColor", "#212529")
        st._config.set_option("theme.font", "monospace")
        st.session_state.theme = "light"
    else:
        # Switch back to your DARK theme
        st._config.set_option("theme.base", "dark")
        st._config.set_option("theme.primaryColor", "#EF7849")
        st._config.set_option("theme.backgroundColor", "#0A0C0F")
        st._config.set_option("theme.secondaryBackgroundColor", "#12151A")
        st._config.set_option("theme.textColor", "#E7E9EC")
        st._config.set_option("theme.font", "monospace")
        st.session_state.theme = "dark"
    st.rerun()
# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.caption("THEME")
    col_toggle, _, _ = st.columns([1, 1, 1])
    with col_toggle:
        if st.button(st.session_state.theme.upper(), help="Toggle Light/Dark Theme", use_container_width=True):
            toggle_theme()  # Function defined at the top of your script

    mode = mode_selector("MODE", "mode")
    st.markdown("---")
    if mode != "Scale":
        st.caption("Choose preset")
        scenario = st.radio(
            "scenario_radio", list(PRESETS.keys()), label_visibility="collapsed"
        )
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

    if mode == "Comparison":
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
    if mode != "Scale":
        st.markdown("---")

        load_preset_btn = st.button("Load preset ▸", use_container_width=True)
    else:
        load_preset_btn = False

if mode != "Scale":
    # ── Top area ─────────────────────────────────────────────────────────
    preset = PRESETS[scenario]
    query_default = preset["query"]
    if isinstance(query_default, list):
        query_default = query_default[0]

    st.caption(f"CONTEXT · loaded from {scenario}")
    context = st.text_area(
        "context_area",
        value=preset["context"],
        height=120,
        label_visibility="collapsed",
    )

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        st.caption("QUERY")
        query = st.text_input(
            "query_input", value=query_default, label_visibility="collapsed"
        )
    with col_btn:
        st.caption('<span style="visibility:hidden;">Q</span>', unsafe_allow_html=True)
        run_inference_btn = st.button("Run inference ▸", use_container_width=True)

    st.markdown("---")


ARROW = '<div style="text-align:center;color:#555;margin:2px 0;">↓</div>'

def _inputs_match_preset(preset, context, query):
    """True when the user hasn't edited the context or query away from the preset."""
    preset_query = preset["query"]
    if isinstance(preset_query, list):
        preset_query = preset_query[0]
    return context.strip() == preset["context"].strip() and query.strip() == preset_query.strip()


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
        "displayed": "",
        "n_chunks": 0,
        "n_comp": 0,
        "mem_html": "",
        "result": None,
    }
    return phs, state


def _handle_stage(update, phs, state, color, tpc):
    """Process one generator yield and update the corresponding placeholder."""
    stage = update["stage"]
    et = update.get("exec_time")
    if stage == "compression":
        embs = update["compressed_embs"]
        state["compression_time"] = et
        state["n_chunks"] = embs.shape[0]
        state["mem_html"] = _mem_tokens_html(
            [
                embs[i, j].tolist()
                for i in range(embs.shape[0])
                for j in range(embs.shape[1])
            ],
            state["n_chunks"],
            color,
        )
        phs["tokens"].markdown(
            _stage_box(state["mem_html"], active=True, color=color, exec_time=et),
            unsafe_allow_html=True,
        )
        phs["arrow1"].markdown(ARROW, unsafe_allow_html=True)
    elif stage == "router":
        state["n_comp"] = state["n_chunks"] if update["mode"] == "compressed" else 0
        state["router_time"] = et
        state["clf_prob"] = update.get("clf_prob")
        state["route_mode"] = update["mode"]
        inner = _router_chips_html(
            state["n_chunks"],
            state["n_comp"],
            color,
            tpc,
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
            _stage_box(_decoder_html(color), active=True, color=color),
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


def _finalize(phs, state, model, gold, show_gold=True):
    """Deactivate all boxes and render final state."""
    color = COLORS[model.model_name]
    tpc = TOKENS_PER_CHUNK[model.model_name]
    n_chunks, n_comp, mem_html = state["n_chunks"], state["n_comp"], state["mem_html"]
    result = state["result"]

    phs["header"].markdown(
        _header_html(model.model_name, color),
        unsafe_allow_html=True,
    )
    phs["tokens"].markdown(
        _stage_box(mem_html, exec_time=state["compression_time"]),
        unsafe_allow_html=True,
    )
    phs["router"].markdown(
        _stage_box(
            _router_chips_html(
                n_chunks,
                n_comp,
                color,
                tpc,
                clf_prob=state.get("clf_prob"),
                route_mode=state.get("route_mode"),
            ),
            exec_time=state["router_time"],
        ),
        unsafe_allow_html=True,
    )
    phs["decoder"].markdown(_stage_box(_decoder_html(color)), unsafe_allow_html=True)
    if show_gold:
        preset = state.get("preset")
        if preset and result["mode"] == "compressed":
            correct = preset.get("matches_gold", True)
        else:
            correct = True
        pill_var = "badge_green" if correct else "badge_red"
        pill_label = "✓&nbsp;&nbsp;CORRECT" if correct else "✗&nbsp;&nbsp;INCORRECT"
        phs["answer_hdr"].markdown(
            f'<div style="margin-top:8px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:var(--t-text2);font-size:12px;font-weight:bold;">ANSWER</span>'
            f'<span style="background:var(--t-{pill_var}_bg);color:var(--t-{pill_var});font-size:11px;font-weight:600;'
            f'padding:2px 12px;border-radius:6px;">{pill_label}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        phs["answer_hdr"].markdown(
            '<div style="margin-top:8px;">'
            '<span style="color:var(--t-text2);font-size:12px;font-weight:bold;">ANSWER</span></div>',
            unsafe_allow_html=True,
        )
    phs["answer"].markdown(
        f'<div style="font-size:15px;min-height:60px;color:var(--t-answer_text);">{result["prediction"]}</div>',
        unsafe_allow_html=True,
    )
    if show_gold:
        phs["gold"].caption(f"gold · {gold}")
    phs["metrics"].markdown(
        f'<div style="display:flex;gap:24px;margin-top:8px;padding-top:8px;border-top:1px solid var(--t-border1);">'
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
    """Merge per-model result dict with top-level full_answer into one preset dict."""
    if not preset_obj:
        return None
    return preset_obj.get("results", {}).get(model_name)


def run_pipeline_live(
    model, context, query, threshold, container, gold, preset_obj=None, show_gold=True
):
    """Drive a single model's pipeline into Streamlit placeholders."""
    color = COLORS[model.model_name]
    tpc = TOKENS_PER_CHUNK[model.model_name]
    preset = _build_preset(preset_obj, model.model_name)
    phs, state = _setup_column(model, container)
    state["preset"] = preset
    state["threshold"] = threshold

    for update in model.run_pipeline_with_progress(
        context, query, threshold=threshold, preset=preset,
    ):
        _handle_stage(update, phs, state, color, tpc)

    return _finalize(phs, state, model, gold, show_gold=show_gold)


def run_comparison_live(
    m_left,
    m_right,
    context,
    query,
    th_a,
    th_b,
    col_left,
    col_right,
    gold,
    preset_obj=None,
    show_gold=True,
):
    """Drive two models with interleaved token streaming."""
    phs_l, st_l = _setup_column(m_left, col_left)
    phs_r, st_r = _setup_column(m_right, col_right)
    st_l["threshold"] = th_a
    st_r["threshold"] = th_b
    color_l, tpc_l = COLORS[m_left.model_name], TOKENS_PER_CHUNK[m_left.model_name]
    color_r, tpc_r = COLORS[m_right.model_name], TOKENS_PER_CHUNK[m_right.model_name]
    preset_l = _build_preset(preset_obj, m_left.model_name)
    preset_r = _build_preset(preset_obj, m_right.model_name)
    st_l["preset"] = preset_l
    st_r["preset"] = preset_r

    gen_l = m_left.run_pipeline_with_progress(
        context, query, threshold=th_a, preset=preset_l,
    )
    gen_r = m_right.run_pipeline_with_progress(
        context, query, threshold=th_b, preset=preset_r,
    )

    # Interleave all stages — alternate next() calls so both columns animate together
    done_l = done_r = False
    while not done_l or not done_r:
        if not done_l:
            try:
                u = next(gen_l)
                _handle_stage(u, phs_l, st_l, color_l, tpc_l)
                if u["stage"] == "done":
                    done_l = True
            except StopIteration:
                done_l = True
        if not done_r:
            try:
                u = next(gen_r)
                _handle_stage(u, phs_r, st_r, color_r, tpc_r)
                if u["stage"] == "done":
                    done_r = True
            except StopIteration:
                done_r = True

    return _finalize(phs_l, st_l, m_left, gold, show_gold=show_gold), _finalize(phs_r, st_r, m_right, gold, show_gold=show_gold)


SCALE_DATA = json.loads((Path(__file__).parent / "scale_data.json").read_text())


def _scale_metric_card(label, value, sub="", accent=None):
    if accent:
        bg = f"background:linear-gradient(135deg, {accent}18 0%, {accent}08 100%);border:1px solid {accent}30;"
        label_color = accent
        value_color = accent
        sub_color = f"{accent}99"
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
    dark = st.session_state.theme == "dark"
    svg_bg = "#12151a" if dark else "#F1F3F5"
    svg_grid = "#1e2128" if dark else "#ddd"
    svg_text = "#555" if dark else "#999"
    svg_label_light = "#ccc" if dark else "#444"

    svg = (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;display:block;">'
        f'<rect width="{w}" height="{h}" fill="{svg_bg}" rx="8"/>'
    )

    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        yp = sy(yv)
        svg += f'<line x1="{pad_l}" y1="{yp}" x2="{w - pad_r}" y2="{yp}" stroke="{svg_grid}" stroke-width="1"/>'
        svg += f'<text x="{pad_l - 8}" y="{yp + 4}" fill="{svg_text}" font-size="10" text-anchor="end" font-family="monospace">{yv:.0f}%</text>'

    for i in range(7):
        xv = i * 50
        xp = sx(xv)
        svg += f'<line x1="{xp}" y1="{pad_t}" x2="{xp}" y2="{h - pad_b}" stroke="{svg_grid}" stroke-width="1"/>'
        svg += f'<text x="{xp}" y="{h - pad_b + 16}" fill="{svg_text}" font-size="10" text-anchor="middle" font-family="monospace">{xv:.0f}</text>'

    svg += f'<text x="{w // 2}" y="{h - 4}" fill="{svg_text}" font-size="10" text-anchor="middle" font-family="monospace">avg tokens / query</text>'
    svg += f'<text x="12" y="{h // 2}" fill="{svg_text}" font-size="10" text-anchor="middle" font-family="monospace" transform="rotate(-90,12,{h // 2})">Accuracy</text>'

    dot_muted = "#888" if dark else "#777"
    dot_full = "#ccc" if dark else "#444"
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
        svg += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{dc}"/>'
        dx, dy1, dy2, anchor = label_cfg[i]
        svg += f'<text x="{cx + dx}" y="{cy + dy1}" fill="{dc}" font-size="11" text-anchor="{anchor}" font-family="monospace">{lab}</text>'
        svg += f'<text x="{cx + dx}" y="{cy + dy2}" fill="{svg_text}" font-size="9" text-anchor="{anchor}" font-family="monospace">{xv:.0f} tok · {yv:.1f}%</text>'

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
        f'<span style="font-size:22px;font-weight:800;color:var(--t-text1);">{model_name}</span>'
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
    comp_correct = 0
    full_correct = 0
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
        pct = done / n


        ph_progress.markdown(
            f'<div style="background:var(--t-progress_outer);border-radius:8px;padding:8px 12px;margin:8px 0;">'
            f'<div style="color:var(--t-text2);font-size:11px;margin-bottom:4px;">{done}/{n} samples</div>'
            f'<div style="background:var(--t-progress_bg);border-radius:4px;height:12px;overflow:hidden;">'
            f'<div style="background:{color};height:100%;width:{pct * 100:.1f}%;'
            f'border-radius:4px;transition:width 0.05s;"></div></div></div>',
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
            f'<div style="display:flex;gap:12px;margin:8px 0;">'
            + _scale_metric_card(
                "Tokens saved",
                f"{saved_pct:.1f}%",
                f"{total_tokens_saved:,} / {total_tokens_full:,}",
                accent="#2ea84a" if st.session_state.theme == "light" else "#66ff7f",
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

        time.sleep(0.01)


# ── Main layout ──────────────────────────────────────────────────────


if mode == "Scale":
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        _run_scale_page(model_a, threshold_a)

elif mode == "Comparison":
    if load_preset_btn:
        st.session_state["context_area"] = preset["context"]
        st.session_state["query_input"] = query_default
    do_run = load_preset_btn or run_inference_btn
    if do_run:
        show_gold = _inputs_match_preset(preset, context, query)
        col_left, col_right = st.columns(2)
        m_left = MockModel(model_a)
        try:
            with _GPU_LOCK:
                _ensure_gpu(model_b)
                m_right = load_model(model_b)
                left_result, right_result = run_comparison_live(
                    m_left,
                    m_right,
                    context,
                    query,
                    threshold_a,
                    threshold_b,
                    col_left,
                    col_right,
                    preset["gold"],
                    preset_obj=preset,
                    show_gold=show_gold,
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
                preset["gold"],
                preset_obj=preset,
                show_gold=show_gold,
            )

else:
    if load_preset_btn:
        st.session_state["context_area"] = preset["context"]
        st.session_state["query_input"] = query_default
    do_run = load_preset_btn or run_inference_btn
    if do_run:
        show_gold = _inputs_match_preset(preset, context, query)
        (col_center,) = st.columns([1])
        try:
            with _GPU_LOCK:
                _ensure_gpu(model_a)
                model = load_model(model_a)
                run_pipeline_live(
                    model,
                    context,
                    query,
                    threshold_a,
                    col_center,
                    preset["gold"],
                    preset_obj=preset,
                    show_gold=show_gold,
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
                preset["gold"],
                preset_obj=preset,
                show_gold=show_gold,
            )
