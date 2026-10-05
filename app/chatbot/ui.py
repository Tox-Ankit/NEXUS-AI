import streamlit as st
from app.llm.service import LLMService, OLLAMA_MODEL
from app.chatbot.controller import AIController

def render_chat_panel():
    """
    Renders the NEXUS AI Chat Assistant as a modern, dedicated chatting box.
    Uses Black, Red, and White styling with quick prompt pills, live cloud badge,
    and verified calculation indicators.
    """
    
    # Chat Box Header Card
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #161822, #101117);
            border: 1px solid rgba(255, 51, 75, 0.35);
            border-radius: 12px 12px 0 0;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0px;
        ">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="
                    width: 34px; height: 34px;
                    border-radius: 8px;
                    background: linear-gradient(135deg, #FF334B, #990014);
                    display: flex; align-items: center; justify-content: center;
                    font-size: 18px; box-shadow: 0 0 12px rgba(255, 51, 75, 0.4);
                ">💬</div>
                <div>
                    <div style="color: #FFFFFF; font-weight: 700; font-size: 14px; letter-spacing: 0.5px;">NEXUS AI ASSISTANT</div>
                    <div style="display: flex; align-items: center; gap: 6px; font-size: 11px; color: #94A3B8;">
                        <span style="display: inline-block; width: 7px; height: 7px; background: #22C55E; border-radius: 50%; box-shadow: 0 0 6px #22C55E;"></span>
                        <span>Ollama Cloud · <b>{OLLAMA_MODEL}</b></span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    st.markdown(
        '<div style="font-size:11px;color:#94A3B8;font-weight:600;margin:8px 0 6px 0;">QUICK PROMPTS</div>',
        unsafe_allow_html=True,
    )
    row1 = st.columns(4)
    with row1[0]:
        if st.button("📊 Overview", key="pill_overview", use_container_width=True):
            st.session_state.pending_prompt = "What is the overall summary of this dataset?"
    with row1[1]:
        if st.button("📈 Trends", key="pill_trends", use_container_width=True):
            st.session_state.pending_prompt = "What trends or patterns stand out in the numeric columns?"
    with row1[2]:
        if st.button("🔮 Best Model", key="pill_model", use_container_width=True):
            st.session_state.pending_prompt = "Which predictive model performed best and what are the metrics?"
    with row1[3]:
        if st.button("🗑️ Reset", key="clear_chat_btn", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Check LLM Availability
    is_llm_available = LLMService.is_available()
    if not is_llm_available:
        st.error("⚠️ AI is currently unavailable. Please check your OLLAMA_API_KEY in .env.")
        return

    # Chat Messages Window Container
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style="
                    text-align: center;
                    padding: 30px 15px;
                    color: #94A3B8;
                    background: rgba(22, 24, 34, 0.4);
                    border: 1px dashed rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    margin: 10px 0;
                ">
                    <div style="font-size: 32px; margin-bottom: 8px;">🧠</div>
                    <div style="color: #FFFFFF; font-weight: 600; margin-bottom: 4px;">Welcome to NEXUS Chat Studio</div>
                    <div style="font-size: 12px; line-height: 1.5;">
                        Ask natural language questions about your data, metrics, or machine learning models.<br>
                        <span style="color: #FF4D6D;">All calculations are verified by Python & DuckDB.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for message in st.session_state.messages:
                role = message["role"]
                with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🧠"):
                    st.markdown(message["content"])
                    if role == "assistant" and message.get("verified", True):
                        st.markdown(
                            "<div style='font-size: 10px; color: #FF4D6D; margin-top: 4px;'>🛡️ Verified Engine Calculations</div>",
                            unsafe_allow_html=True
                        )

    # Check for pending prompt from pills
    queued_prompt = None
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        queued_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    # Chat Input Box
    placeholder_text = "Ask anything about your data or models..." if "dataset" in st.session_state else "Please upload a dataset first..."
    user_input = st.chat_input(placeholder_text, key="chat_user_input")
    
    active_prompt = queued_prompt or user_input
    
    if active_prompt:
        # Check if dataset is loaded
        if "dataset" not in st.session_state:
            bot_response = "Please upload a dataset from the sidebar first, and I will immediately analyze it for you!"
            st.session_state.messages.append({"role": "user", "content": active_prompt})
            st.session_state.messages.append({"role": "assistant", "content": bot_response, "verified": False})
            st.rerun()

        # Append user message
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        
        with chat_container:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(active_prompt)

            with st.chat_message("assistant", avatar="🧠"):
                placeholder = st.empty()
                with st.spinner("Analyzing and computing with Ollama Cloud..."):
                    try:
                        profile_ctx = st.session_state.get("profile", {})
                        stats_ctx = st.session_state.get("stats", {})
                        ml_ctx = st.session_state.get("ml_result", None)
                        parquet_path = st.session_state.get("parquet_path", None)

                        response = AIController.process_query(
                            query=active_prompt,
                            dataset_context=profile_ctx,
                            stats_context=stats_ctx,
                            ml_context=ml_ctx,
                            parquet_path=parquet_path
                        )
                        
                        placeholder.markdown(response)
                        st.markdown(
                            "<div style='font-size: 10px; color: #FF4D6D; margin-top: 4px;'>🛡️ Verified Engine Calculations</div>",
                            unsafe_allow_html=True
                        )
                        st.session_state.messages.append({"role": "assistant", "content": response, "verified": True})
                    except Exception as e:
                        err_msg = f"Sorry, I encountered an error while analyzing: {str(e)}"
                        placeholder.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg, "verified": False})
        
        st.rerun()
