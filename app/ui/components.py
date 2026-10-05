import streamlit as st


def section_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<p class="nexus-section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="nexus-section-title">{title}</div>{sub_html}',
        unsafe_allow_html=True,
    )


def sidebar_heading(title: str) -> None:
    st.markdown(
        f'<div class="sidebar-section-title" style="color:#FF334B;font-size:15px;font-weight:700;margin:8px 0;">{title}</div>',
        unsafe_allow_html=True,
    )


def workflow_steps(active: str) -> None:
    steps = [
        ("load", "① Load"),
        ("profile", "② Profile"),
        ("visualize", "③ Visualize"),
        ("predict", "④ Predict"),
        ("report", "⑤ Report"),
    ]
    chips = []
    for key, label in steps:
        cls = "nexus-workflow-step active" if key == active else "nexus-workflow-step"
        chips.append(f'<span class="{cls}">{label}</span>')
    st.markdown(f'<div class="nexus-workflow">{"".join(chips)}</div>', unsafe_allow_html=True)


def chart_insight(text: str) -> None:
    if text:
        st.markdown(f'<div class="nexus-insight">{text}</div>', unsafe_allow_html=True)
