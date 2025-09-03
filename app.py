import streamlit as st
from rag_utils import generate_answer

# --- Page config ---
st.set_page_config(page_title="GenAI Q&A", page_icon="📚🤖")
st.title("📚🤖 GenAI Assistant")
st.caption("Your smart Q&A powered by LLM + RAG")

# --- Session state setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [(question, answer)]

# --- Chat input ---
if user_input := st.chat_input("Enter your question?"):
    # Append placeholder entry for the new question
    st.session_state.chat_history.append((user_input, None))

    # Rerun to refresh UI
    st.rerun()

# --- Display conversation ---
for idx, (question, answer) in enumerate(st.session_state.chat_history):
    st.subheader(f"🙍 : {question}")

    if answer is None:
        # Placeholder for thinking...
        response_placeholder = st.empty()
        response_placeholder.markdown("📚🤖 : Thinking...")

        # Generate the answer
        response = generate_answer(question)

        # Replace placeholder with real response
        st.session_state.chat_history[idx] = (question, response)

        # Rerun again to update UI
        st.rerun()
    else:
        st.markdown(f"📚🤖 : {answer}")
