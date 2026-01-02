import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# --- Initialize tools ---
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchResults(name="Search")

# --- Streamlit UI ---
st.title("🔎 LangChain - Chat with Arxiv, Wikipedia & Web Search")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input(placeholder="What do you want to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        streaming=True
    )

    tools = [search, arxiv, wiki]

    # Initialize agent with max 5 iterations
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=5,
        verbose=True,
        early_stopping_method="generate"  # ✅ Try to generate best final answer even if stopped early
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            # ✅ In case it hits iteration/time limit, extract partial result
            if "Agent stopped due to iteration limit" in str(e):
                response = "⚠️ I reached my 5-step search limit but here’s what I found so far."
            else:
                response = f"❌ Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
