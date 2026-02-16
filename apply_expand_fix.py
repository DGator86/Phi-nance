#!/usr/bin/env python3
"""One-shot fix for StreamlitValueAssignmentNotAllowedError on expand button. Run from repo root: python apply_expand_fix.py"""

from pathlib import Path

APP = Path(__file__).resolve().parent / "app_v2.py"

OLD = """    with col3:
        expand_key = f"expand_{strategy['id']}"
        if expand_key not in st.session_state:
            st.session_state[expand_key] = False
        expanded = st.button("⚙️", key=expand_key, help="Configure parameters")
        if expanded:
            st.session_state[expand_key] = not st.session_state[expand_key]
    
    if enabled:
        enabled_strategies.append(strategy["id"])
    
    # Expandable parameters section
    if st.session_state.get(expand_key, False):"""

NEW = """    with col3:
        expand_btn_key = f"expand_btn_{strategy['id']}"
        expanded_state_key = f"_expanded_{strategy['id']}"
        if expanded_state_key not in st.session_state:
            st.session_state[expanded_state_key] = False
        if st.button("⚙️", key=expand_btn_key, help="Configure parameters"):
            st.session_state[expanded_state_key] = not st.session_state[expanded_state_key]
    
    if enabled:
        enabled_strategies.append(strategy["id"])
    
    # Expandable parameters section
    if st.session_state.get(expanded_state_key, False):"""

def main():
    if not APP.exists():
        print("app_v2.py not found. Run from repo root.")
        return 1
    text = APP.read_text(encoding="utf-8")
    if OLD not in text and NEW in text:
        print("app_v2.py already has the fix. Nothing to do.")
        return 0
    if OLD not in text:
        print("Could not find the exact block to replace. Apply the fix from FIX_EXPAND_BUTTON.md manually.")
        return 1
    APP.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("Applied expand-button fix to app_v2.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
