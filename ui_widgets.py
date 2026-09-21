"""
ui_widgets.py — кастомные Streamlit-виджеты: карточки метрик и
слайдеры/select_slider с кнопками +/- по бокам (через session_state + callback).
"""
import streamlit as st


def metric_card(label, value, variant=None):
    extra_class = f" {variant}" if variant else ""
    st.markdown(
        f'<div class="metric-card-sm"><div class="metric-label-sm">{label}</div>'
        f'<div class="metric-value-sm{extra_class}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def dec(key, min_val):
    st.session_state[key] = max(min_val, st.session_state[key] - 1)


def inc(key, max_val):
    st.session_state[key] = min(max_val, st.session_state[key] + 1)


def slider_with_buttons(label, key, min_val, max_val, default_val, step=1, help_text=None):
    if key not in st.session_state:
        st.session_state[key] = default_val
    st.session_state[key] = max(min_val, min(st.session_state[key], max_val))
    col_minus, col_slider, col_plus = st.columns([1, 8, 1])
    with col_minus:
        st.button("➖", key=f"{key}_minus_btn", on_click=dec, args=(key, min_val))
    with col_slider:
        st.slider(label, min_val, max_val, help=help_text, key=key, step=step)
    with col_plus:
        st.button("➕", key=f"{key}_plus_btn", on_click=inc, args=(key, max_val))
    return st.session_state[key]


def select_slider_with_buttons(label, key, options_list, help_text=None):
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] = min(st.session_state[key], len(options_list) - 1)
    col_minus, col_slider, col_plus = st.columns([1, 8, 1])
    with col_minus:
        st.button("➖", key=f"{key}_minus_btn", on_click=dec, args=(key, 0))
    with col_slider:
        st.select_slider(label, options=range(len(options_list)),
                          format_func=lambda i: options_list[i], help=help_text, key=key)
    with col_plus:
        st.button("➕", key=f"{key}_plus_btn", on_click=inc, args=(key, len(options_list) - 1))
    return st.session_state[key]
