import os
import streamlit as st
import torch
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig  # For 8-bit quantization

# Set Streamlit page config
st.set_page_config(
    page_title="Bulipe Tech Assistant",
    page_icon="icon.png",
    layout="wide"
)

# --- Apply Custom Style ---
def set_custom_style(background_image_path):
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

set_custom_style("R.jpg")  # Your custom background image

# --- Load Qwen2-0.5B-Instruct model ---
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2-0.5B-Instruct"  # Switched to smaller model for Streamlit Cloud compatibility
    # Use 8-bit quantization to reduce memory usage
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

# --- System Prompt ---
system_message = {
    "role": "system",
    "content": "You are the official AI assistant of Bulipe Tech, developed by Mohammod Ibrahim Hossain. Your primary function is to support and inform customers by providing accurate and detailed information about the company's products, services, and training programs. You are designed to deliver clear, helpful, and technically sound assistance to enhance customer understanding and engagement with Bulipe Tech's offerings.\n\nBulipe Tech currently offers the following professional training programs:\n\n1. IT Support Specialist:\n   - Technology maintenance\n   - Troubleshooting techniques\n   - Client interaction skills\n\n2. Digital Marketing:\n   - SEO\n   - Social Media Marketing\n   - Email campaigns\n   - Analytics\n   - Plus: Microsoft Office + spoken English\n\n3. Online Sales and Marketing:\n   - Sales techniques\n   - Digital ads\n   - CRM\n   - Plus: Office + spoken English\n\n4. Social Media Specialist:\n   - Content creation\n   - Audience engagement\n   - Platform strategies\n   - Plus: Office + spoken English\n\n5. Online Posting Specialist:\n   - Posting strategies\n   - Scheduling\n   - Platform knowledge\n   - Plus: Office + spoken English\n\n6. Data Entry & Virtual Assistance:\n   - Data accuracy\n   - Virtual tasks\n   - Time management\n   - Plus: Office + spoken English"
}

# --- Sidebar ---
with st.sidebar:
    st.title("Chat Settings")
    def clear_chat():
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Assalamu alaikum 🍁, I'm your AI assistant from Bulipe Tech. How can I help you today? 😊"
        }]
    st.button("Clear Chat", on_click=clear_chat)

# --- Initialize chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Assalamu alaikum 🍁, I'm your AI assistant from Bulipe Tech. How can I help you today? 😊"
    }]

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Generate response ---
def generate_response(prompt):
    messages = [system_message] + [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1
    )
    generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response

# --- Handle new input ---
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            answer = generate_response(user_input)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})