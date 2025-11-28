import streamlit as st
from src.graph import build_graph
from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Weather & Style Agent", page_icon="🌤️")
st.title("🌤️ Weather & Style Agent")
st.markdown("Ask me about the weather or what to wear!")

# --- CSS FOR CHAT  ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE GRAPH ---
# cache the graph 
@st.cache_resource
def get_graph():
    return build_graph()

app = get_graph()

# --- SESSION STATE (Chat History) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY HISTORY ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# --- CHAT INPUT & LOGIC ---
if user_input := st.chat_input("What's on your mind?"):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 2. Run Graph (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # We create a placeholder to show "Thinking..." steps
        status_placeholder = st.status("Thinking...", expanded=True)
        
        try:
            # We use .stream() to see steps happen in real-time
            input_structure = {"messages": st.session_state.messages}
            config = {"recursion_limit": 5}
            
            for event in app.stream(input_structure, config=config):
                
                # Check which node just finished
                if "classifier" in event:
                    data = event["classifier"]
                    status_placeholder.write(f"🧭 **Router:** Intent detected as `{data['intent']}`")
                    
                elif "weather_node" in event:
                    data = event["weather_node"]
                    # Show a snippet of the raw data
                    preview = data['context'][:100] + "..." if len(data['context']) > 100 else data['context']
                    status_placeholder.write(f"☁️ **Weather Tool:** Retrieved data for city.\n`{preview}`")
                    
                elif "rag_node" in event:
                    data = event["rag_node"]
                    status_placeholder.write(f"📚 **RAG Tool:** Searched vector store.\n*Context Found*")
                    
                elif "answer_node" in event:
                    # This is the final answer
                    ai_msg = event["answer_node"]["messages"][-1]
                    full_response = ai_msg.content
                    
            # Update status to complete
            status_placeholder.update(label="Done!", state="complete", expanded=False)
            
            # Show final answer
            message_placeholder.markdown(full_response)
            
            # Save AI response to history
            st.session_state.messages.append(AIMessage(content=full_response))

        except Exception as e:
            status_placeholder.update(label="Error", state="error")
            st.error(f"An error occurred: {e}")