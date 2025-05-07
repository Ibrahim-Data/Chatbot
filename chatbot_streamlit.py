import os
import streamlit as st
import torch
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import streamlit.watcher.local_sources_watcher as lsw

# Patch LocalSourcesWatcher to avoid torch.classes error
def patched_extract_paths(module):
    try:
        return list(module.__path__._path)
    except AttributeError:
        return []

lsw.extract_paths = lambda m: patched_extract_paths(m)

# Set Streamlit page config
st.set_page_config(
    page_title="Bulipe Tech Assistant",
    page_icon="icon.png",
    layout="wide"
)

# Apply Custom Style
def set_custom_style(background_image_path):
    try:
        with open(background_image_path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stSidebar .sidebar-content {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(5px);
            padding: 1rem;
            border-radius: 10px;
        }}
        .main .block-container {{
            padding: 2rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Background image not found. Using default styling.")
        css = """
        <style>
        .stSidebar .sidebar-content {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(5px);
            padding: 1rem;
            border-radius: 10px;
        }}
        .main .block-container {{
            padding: 2rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

set_custom_style("R.jpg")  # Your custom background image

# Load Qwen2-0.5B-Instruct model
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# System Prompt
system_message = {
    "role": "system",
    "content": """You are the official AI assistant of Bulipe Tech, developed by Mohammod Ibrahim Hossain. Your primary role is to provide accurate, detailed, and clear information about Bulipe Tech's services and training programs. When a user asks about services, programs, or anything related to Bulipe Tech's offerings, always respond with a structured summary of the relevant training programs, including key details. If the query is unrelated, provide a helpful and concise response.

Bulipe Tech offers the following professional training programs:

1. **IT Support Specialist**:
   - Technology maintenance
   - Troubleshooting techniques
   - Client interaction skills

2. **Digital Marketing**:
   - SEO
   - Social Media Marketing
   - Email campaigns
   - Analytics
   - Additional: Microsoft Office + spoken English

3. **Online Sales and Marketing**:
   - Sales techniques
   - Digital ads
   - CRM
   - Additional: Office + spoken English

4. **Social Media Specialist**:
   - Content creation
   - Audience engagement
   - Platform strategies
   - Additional: Office + spoken English

5. **Online Posting Specialist**:
   - Posting strategies
   - Scheduling
   - Platform knowledge
   - Additional: Office + spoken English

6. **Data Entry & Virtual Assistance**:
   - Data accuracy
   - Virtual tasks
   - Time management
   - Additional: Office + spoken English"""
}

# Sidebar
with st.sidebar:
    st.title("Chat Settings")
    def clear_chat():
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Assalamu alaikum 🍁, I'm your AI assistant from Bulipe Tech. How can I help you today? 😊 Ask about our training programs or services!"
        }]
    st.button("Clear Chat", on_click=clear_chat)

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Assalamu alaikum 🍁, I'm your AI assistant from Bulipe Tech. How can I help you today? 😊 Ask about our training programs or services!"
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Generate response
def generate_response(prompt):
    service_keywords = ["service", "services", "training", "program", "programs", "course", "courses", "offering", "offerings"]
    is_service_query = any(keyword in prompt.lower() for keyword in service_keywords)

    if is_service_query:
        user_message = {
            "role": "user",
            "content": f"{prompt}\n\nPlease provide a detailed summary of Bulipe Tech's training programs, including all relevant details about each program."
        }
    else:
        user_message = {"role": "user", "content": prompt}

    messages = [system_message, user_message]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1
    )
    generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if is_service_query and not any(keyword in response.lower() for keyword in service_keywords):
        response += "\n\nHere is a summary of Bulipe Tech's training programs:\n\n" + system_message["content"].split("Bulipe Tech offers")[1]

    return response

# Handle new input
if user_input := st.chat_input("Ask me anything..."):
    modified_input = user_input.replace("you", "Bulipe Tech").replace("You", "Bulipe Tech")
    st.session_state.messages.append({"role": "user", "content": modified_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            answer = generate_response(modified_input)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})