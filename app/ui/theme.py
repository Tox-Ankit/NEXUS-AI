import streamlit as st


def inject_nexus_theme() -> None:
    """Inject NEXUS design system (typography, layout, tabs, buttons, cards)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #0A0B0E;
            color: #FFFFFF;
        }

        section[data-testid="stSidebar"] {
            background-color: #101117 !important;
            border-right: 1px solid rgba(255, 51, 75, 0.2) !important;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .sidebar-section-title {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #12131C;
            padding: 8px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94A3B8;
            font-weight: 600;
            padding: 8px 16px;
            background-color: transparent;
            transition: all 0.2s ease-in-out;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #FF334B, #B80C23) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(255, 51, 75, 0.35);
        }

        div.stButton > button:first-child {
            background: linear-gradient(135deg, #FF334B, #C70A24);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.25s ease-in-out;
            box-shadow: 0 4px 12px rgba(255, 51, 75, 0.25);
        }

        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #FF4D6D, #E50914);
            box-shadow: 0 6px 18px rgba(255, 51, 75, 0.45);
            transform: translateY(-1px);
            color: #FFFFFF;
        }

        [data-testid="stMetricValue"] {
            color: #FF334B !important;
            font-weight: 700 !important;
            font-size: 1.65rem !important;
        }

        [data-testid="stMetricLabel"] {
            color: #E2E8F0 !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }

        .stDataFrame {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
        }

        [data-testid="stChatMessage"] {
            background-color: #12141D !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 12px !important;
            margin-bottom: 8px !important;
        }

        .nexus-card {
            background: #141620;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }

        .nexus-champion-card {
            background: linear-gradient(135deg, rgba(255, 51, 75, 0.12), rgba(18, 20, 29, 0.9));
            border: 1px solid #FF334B;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(255, 51, 75, 0.15);
        }

        .nexus-section-title {
            font-size: 17px;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0 0 4px 0;
            letter-spacing: 0.2px;
        }

        .nexus-section-sub {
            font-size: 13px;
            color: #94A3B8;
            margin: 0 0 14px 0;
            line-height: 1.45;
        }

        .nexus-insight {
            font-size: 12px;
            color: #CBD5E1;
            background: rgba(255, 51, 75, 0.08);
            border-left: 3px solid #FF334B;
            padding: 10px 12px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0 16px 0;
        }

        .nexus-workflow {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 18px;
        }

        .nexus-workflow-step {
            font-size: 11px;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #94A3B8;
            background: #12131C;
        }

        .nexus-workflow-step.active {
            border-color: rgba(255, 51, 75, 0.55);
            color: #FFFFFF;
            background: rgba(255, 51, 75, 0.15);
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stRadio"] label {
            font-weight: 600 !important;
            color: #E2E8F0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
