"""Shared page wrapper used by the one-line scripts in views/."""
import streamlit as st

from dashboard.data import load_all


def render(fn, needs_data=True):
    try:
        if needs_data:
            fn(load_all())
        else:
            fn()
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.exception(exc)
