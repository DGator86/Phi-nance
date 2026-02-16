# Single fix: StreamlitValueAssignmentNotAllowedError (expand button)

**Error:** `Values for the widget with key 'expand_buy_and_hold' cannot be set using st.session_state`

**Fix:** In `app_v2.py`, do **one** find-and-replace.

---

## 1. Find this (delete it):

```python
    with col3:
        expand_key = f"expand_{strategy['id']}"
        if expand_key not in st.session_state:
            st.session_state[expand_key] = False
        expanded = st.button("⚙️", key=expand_key, help="Configure parameters")
        if expanded:
            st.session_state[expand_key] = not st.session_state[expand_key]
    
    if enabled:
        enabled_strategies.append(strategy["id"])
    
    # Expandable parameters section
    if st.session_state.get(expand_key, False):
```

---

## 2. Replace with this:

```python
    with col3:
        expand_btn_key = f"expand_btn_{strategy['id']}"
        expanded_state_key = f"_expanded_{strategy['id']}"
        if expanded_state_key not in st.session_state:
            st.session_state[expanded_state_key] = False
        if st.button("⚙️", key=expand_btn_key, help="Configure parameters"):
            st.session_state[expanded_state_key] = not st.session_state[expanded_state_key]
    
    if enabled:
        enabled_strategies.append(strategy["id"])
    
    # Expandable parameters section
    if st.session_state.get(expanded_state_key, False):
```

---

**One shot:** Use your editor’s Find (Ctrl+F / Cmd+F) with “expand_key = f” to jump to the spot, then replace the whole block above (Find → Replace or select and paste the replacement).
