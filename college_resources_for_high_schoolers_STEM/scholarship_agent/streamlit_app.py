"""Interactive Streamlit chat using the agent's full six-tool set."""
import logging

import streamlit as st  # type: ignore[import-not-found]

from notebook_backend import handle_request, connection_details

logger = logging.getLogger(__name__)

st.set_page_config(page_title="College & Scholarship Q&A")
st.title("College & Scholarship Q&A")
st.caption("Notebook-compatible hybrid retrieval: existing scholarships and colleges namespaces. No re-ingestion.")

with st.sidebar:
    st.subheader("Agent mode")
    st.info("All six tools are available for every question (no tool loadout).")
    st.caption(
        "Questions can trigger Nebius, Pinecone, and—only when enabled—Tavily API calls. "
        "Avoid entering sensitive student information."
    )
    if st.button("Check existing Pinecone connection"):
        try:
            st.json(connection_details())
        except Exception as exc:
            st.error(str(exc))
    if st.button("Clear conversation"):
        st.session_state["chat_messages"] = []
        st.session_state["agent_history"] = []
        st.rerun()

st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("agent_history", [])

for entry in st.session_state["chat_messages"]:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("trace"):
            with st.expander("Tool activity"):
                for item in entry["trace"]:
                    st.markdown(f"**{item['tool']}**")
                    st.write(item["content"])

question = st.chat_input("Ask about colleges, scholarships, eligibility, amounts, or deadlines")
if question:
    st.session_state["chat_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the available sources..."):
                result = handle_request(
                    question,
                    history=st.session_state["agent_history"],
                    use_loadout=False,
                )
            st.session_state["agent_history"] = result["messages"]
            assistant_entry = {
                "role": "assistant",
                "content": result["answer"],
                "trace": result["trace"],
            }
            st.session_state["chat_messages"].append(assistant_entry)
            st.write(result["answer"])
            with st.expander("Tool activity"):
                st.write("Available tools:", ", ".join(result["selected_tools"]))
                if result["trace"]:
                    for item in result["trace"]:
                        st.markdown(f"**{item['tool']}**")
                        st.write(item["content"])
                else:
                    st.write("The agent did not call a tool for this response.")
        except Exception as exc:
            logger.exception("Streamlit chat request failed")
            message = f"Request failed ({type(exc).__name__}): {exc}"
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": message}
            )
            st.error(message)
