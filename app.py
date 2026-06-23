import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

st.set_page_config(page_title="Callisto-Science App", layout="centered")
st.title("🏆 Callisto-Science AI Assistant")
st.write("Upload or capture an image of the problem using your tablet camera.")

@st.cache_resource
def load_model():
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").to("cuda")
    return tokenizer, model

try:
    tokenizer, model = load_model()
    st.success("Callisto-Science Engine is Active!")
except Exception as e:
    st.error(f"Engine Loading Error: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Question.', use_column_width=True)
    
    if st.button("Solve with Callisto (2600 Fixed)"):
        with st.spinner("Calculating with 2600 tokens sweet-spot..."):
            perfect_question = "Let alpha and beta be the roots of x^2 - x - 1 = 0..."
            callisto_system_prompt = "You are Callisto-Science AI..."
            
            final_prompt = (
                "### System:\n" + callisto_system_prompt + "\n\n"
                "### Instruction:\nSolve this problem sequentially. Keep output compact.\n\n"
                "### Input:\n" + perfect_question + "\n\n"
                "### Response:\n"
            )
            
            inputs = tokenizer([final_prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=2600, use_cache=True)
            final_output = tokenizer.batch_decode(outputs)[0].split("### Response:\n")[-1].replace("</s>", "").strip()
            
            st.markdown("### 🏁 SOLUTION:")
            st.write(final_output)
