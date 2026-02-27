#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-nance Device Detection & Responsive Layout Helper
======================================================
Auto-detects web vs mobile via:
  1. JavaScript screen.width injection (primary, reliable)
  2. Streamlit user-agent header parsing (fallback)
  3. CSS media-query driven class injection (always-on)

Exposes a DeviceInfo dataclass used by dashboard.py and live_workbench.py
to adapt column counts, chart heights, sidebar state, font sizes, etc.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import streamlit as st


class DeviceType(Enum):
    PHONE = "phone"       # <= 480px
    TABLET = "tablet"     # 481-1024px
    DESKTOP = "desktop"   # > 1024px


@dataclass
class DeviceInfo:
    """Resolved device info for the current session."""
    device_type: DeviceType
    screen_width: int
    is_mobile: bool       # phone or tablet
    is_phone: bool        # phone only
    is_tablet: bool       # tablet only
    is_desktop: bool      # desktop only

    # Layout helpers
    cols_kpi: int          # number of KPI cards per row
    cols_strategy: int     # strategy cards per row
    cols_ml: int           # ML model cards per row
    cols_form: int         # form fields per row
    chart_height: int      # default chart height px
    chart_height_sm: int   # small chart height px
    chart_height_lg: int   # large chart height px
    sidebar_state: str     # "expanded" or "collapsed"
    hero_title_size: str   # CSS font-size for hero title
    kpi_font_size: str     # CSS font-size for KPI value


# ---------------------------------------------------------------------------
# JavaScript-based detection (runs once per session)
# ---------------------------------------------------------------------------
_JS_DETECT = """
<script>
(function() {
    // This script runs inside a components.html iframe — target the parent page
    var doc = window.parent.document;
    var win = window.parent;
    var w = win.innerWidth || doc.documentElement.clientWidth || screen.width;
    var h = win.innerHeight || doc.documentElement.clientHeight || screen.height;
    var isTouchDevice = ('ontouchstart' in win) || (navigator.maxTouchPoints > 0);
    var ua = navigator.userAgent || '';
    var isMobileUA = /Mobi|Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua);

    // Determine device type
    var deviceType = 'desktop';
    if (w <= 480 || (isMobileUA && w <= 768)) {
        deviceType = 'phone';
    } else if (w <= 1024 || (isTouchDevice && w <= 1024)) {
        deviceType = 'tablet';
    }

    // Set data attributes on parent body for CSS targeting
    doc.body.setAttribute('data-phi-device', deviceType);
    doc.body.setAttribute('data-phi-width', w);
    doc.body.setAttribute('data-phi-touch', isTouchDevice ? 'true' : 'false');

    // CSS custom properties on parent
    doc.documentElement.style.setProperty('--phi-screen-width', w + 'px');
    doc.documentElement.style.setProperty('--phi-screen-height', h + 'px');

    // Force-close sidebar on mobile
    if (deviceType === 'phone' || deviceType === 'tablet') {
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar && deviceType === 'phone') {
            sidebar.style.display = 'none';
            sidebar.style.width = '0';
        }
        var collapseBtn = doc.querySelector('button[aria-label="Close sidebar"]');
        if (collapseBtn && sidebar && sidebar.getAttribute('aria-expanded') === 'true') {
            collapseBtn.click();
        }
    }

    // Load Material Symbols font into parent document (for expander icons)
    if (!doc.getElementById('phi-material-font')) {
        var link = doc.createElement('link');
        link.id = 'phi-material-font';
        link.rel = 'stylesheet';
        link.href = 'https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap';
        doc.head.appendChild(link);
    }
})();
</script>
"""


def _inject_device_js():
    """Inject the JavaScript device detection via zero-height iframe."""
    import streamlit.components.v1 as components
    components.html(_JS_DETECT, height=0)


# ---------------------------------------------------------------------------
# User-Agent Parsing Fallback
# ---------------------------------------------------------------------------
def _parse_user_agent() -> Optional[str]:
    """Try to detect device from Streamlit's server headers."""
    try:
        # Streamlit 1.37+ uses st.context.headers (non-deprecated)
        headers = st.context.headers
        if headers:
            ua = headers.get("User-Agent", "")
            if any(k in ua.lower() for k in ("iphone", "android", "mobile", "ipod", "webos", "blackberry")):
                return "phone"
            if any(k in ua.lower() for k in ("ipad", "tablet")):
                return "tablet"
            return "desktop"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Resolution Logic
# ---------------------------------------------------------------------------
def _resolve_device(width: Optional[int] = None, ua_hint: Optional[str] = None) -> DeviceType:
    """Resolve device type from available signals."""
    if width is not None:
        if width <= 480:
            return DeviceType.PHONE
        elif width <= 1024:
            return DeviceType.TABLET
        else:
            return DeviceType.DESKTOP

    if ua_hint:
        return DeviceType(ua_hint)

    return DeviceType.DESKTOP  # safe default


def _build_device_info(device: DeviceType, width: int) -> DeviceInfo:
    """Build the full DeviceInfo with all layout helpers."""
    is_phone = device == DeviceType.PHONE
    is_tablet = device == DeviceType.TABLET
    is_desktop = device == DeviceType.DESKTOP
    is_mobile = is_phone or is_tablet

    return DeviceInfo(
        device_type=device,
        screen_width=width,
        is_mobile=is_mobile,
        is_phone=is_phone,
        is_tablet=is_tablet,
        is_desktop=is_desktop,
        # Column counts
        cols_kpi=2 if is_phone else (3 if is_tablet else 4),
        cols_strategy=1 if is_phone else (2 if is_tablet else 4),
        cols_ml=1 if is_phone else (2 if is_tablet else 4),
        cols_form=1 if is_phone else (2 if is_tablet else 3),
        # Chart heights
        chart_height=280 if is_phone else (340 if is_tablet else 400),
        chart_height_sm=200 if is_phone else (250 if is_tablet else 300),
        chart_height_lg=340 if is_phone else (400 if is_tablet else 480),
        # Sidebar
        sidebar_state="collapsed" if is_mobile else "expanded",
        # Typography
        hero_title_size="1.6rem" if is_phone else ("2.2rem" if is_tablet else "3.2rem"),
        kpi_font_size="1.1rem" if is_phone else ("1.3rem" if is_tablet else "1.5rem"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_device(skip_js: bool = False) -> DeviceInfo:
    """
    Detect the user's device and return a DeviceInfo object.
    
    Call this once at the top of main() after set_page_config.
    
    Args:
        skip_js: If True, skip JS injection (caller already injected it).
    
    Usage:
        dev = detect_device()
        if dev.is_mobile:
            cols = st.columns(dev.cols_kpi)
        fig_height = dev.chart_height
    """
    # Inject JS detection (for CSS data-attributes) unless caller already did
    if not skip_js:
        _inject_device_js()

    # Get width from session state (cached from previous run)
    width = st.session_state.get("_phi_screen_width")

    # Fallback: user-agent hint (reliable on Streamlit 1.37+ via st.context.headers)
    ua_hint = _parse_user_agent()

    # Resolve
    device = _resolve_device(width, ua_hint)
    w = width or (375 if device == DeviceType.PHONE else (768 if device == DeviceType.TABLET else 1440))

    info = _build_device_info(device, w)

    # Store in session state for reuse
    st.session_state["_phi_device_info"] = info

    return info


def get_device() -> DeviceInfo:
    """Get cached device info (must call detect_device() first)."""
    info = st.session_state.get("_phi_device_info")
    if info is None:
        return detect_device()
    return info


def inject_responsive_meta():
    """Inject the viewport meta tag for proper mobile rendering."""
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """, unsafe_allow_html=True)
