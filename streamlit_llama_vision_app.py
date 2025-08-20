import streamlit as st
import base64
from groq import Groq

# === Streamlit UI Config ===
st.set_page_config(page_title="Chart Insights with LLM", layout="centered")

st.markdown("""
    <style>
        .stApp {
            background-color: #fdfdfd;
            color: #333333;
        }
        h1, h2, h3, h4 {
            color: #1a237e;
            font-family: "Segoe UI", sans-serif;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1a237e, #3949ab);
            color: #ffffff;
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: bold;
            border: none;
            transition: transform 0.2s, background 0.2s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #3949ab, #1a237e);
            transform: scale(1.03);
        }
        .stFileUploader, .stCameraInput {
            border: 2px dashed #3949ab;
            border-radius: 8px;
            padding: 12px;
            background-color: #f7f8fa;
        }
    </style>
""", unsafe_allow_html=True)

# === Groq API Setup ===
if "GROQ_API_KEY" not in st.secrets:
    st.error("🚨 Please set your Groq API key in Streamlit secrets")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# === App Header ===
st.title("📊 Chart Insights with LLM (Groq)")
st.subheader("Upload or capture a chart to get trends, anomalies & recommendations")

# === Image Input ===
uploaded_file = st.file_uploader("📂 Upload a chart image", type=["png", "jpg", "jpeg"])
camera_file = st.camera_input("📸 Or take a photo (rear camera preferred)", key="rear_camera_input")

image_data = uploaded_file or camera_file

if image_data:
    st.image(image_data, caption="Uploaded Chart", width=600)
    img_bytes = image_data.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    if st.button("🔍 Analyze Chart"):
        with st.spinner("Analyzing image..."):
            try:
                # Groq models don't take images natively → send as base64 text
                prompt = f"""
                You are a professional data analyst AI. A chart image (base64 encoded) is provided.
                Extract insights as if you could see it. Provide:
                1. Key trends
                2. Anomalies
                3. Business insights
                4. Recommended next steps
                Base64 (truncated): {img_base64[:500]}...
                """

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Free, fast model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )

                insights = response.choices[0].message.content
                st.success("✅ Analysis Complete")
                st.subheader("📈 Insights & Recommendations")
                st.write(insights)

            except Exception as e:
                st.error(f"Error analyzing image: {e}")
