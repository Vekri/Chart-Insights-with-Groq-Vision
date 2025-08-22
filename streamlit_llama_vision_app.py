import streamlit as st
import base64
from openai import OpenAI
from groq import Groq

# === Streamlit UI Config ===
st.set_page_config(page_title="Chart Insights App", layout="centered")

st.markdown("""
    <style>
        .stApp {background-color: #fdfdfd; color: #333333;}
        h1, h2, h3, h4 {color: #1a237e; font-family: "Segoe UI", sans-serif;}
        .stButton>button {
            background: linear-gradient(90deg, #1a237e, #3949ab);
            color: #ffffff; border-radius: 8px; padding: 8px 20px;
            font-weight: bold; border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #3949ab, #1a237e);
            transform: scale(1.03);
        }
        .stFileUploader, .stCameraInput {
            border: 2px dashed #3949ab; border-radius: 8px;
            padding: 12px; background-color: #f7f8fa;
        }
    </style>
""", unsafe_allow_html=True)

# === API Setup ===
openai_client, groq_client = None, None

if st.secrets.get("OPENAI_API_KEY") and st.secrets["OPENAI_API_KEY"].startswith("sk-"):
    try:
        openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.warning(f"⚠️ OpenAI init failed: {e}")

if st.secrets.get("GROQ_API_KEY") and st.secrets["GROQ_API_KEY"].startswith("gsk_"):
    try:
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception as e:
        st.warning(f"⚠️ Groq init failed: {e}")

if not (openai_client or groq_client):
    st.error("🚨 No valid API key found. Please set OPENAI_API_KEY and/or GROQ_API_KEY in secrets.")
    st.stop()

# === App Header ===
st.title("📊 Chart Insights")
st.subheader("Upload or capture a chart → Get trends, anomalies & recommendations")

# === Image Input ===
uploaded_file = st.file_uploader("📂 Upload a chart image", type=["png", "jpg", "jpeg"])
camera_file = st.camera_input("📸 Or take a photo", key="rear_camera_input")
image_data = uploaded_file or camera_file

if image_data:
    st.image(image_data, caption="Chart", width=600)
    img_bytes = image_data.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    if st.button("🔍 Analyze Chart"):
        with st.spinner("Analyzing image..."):
            insights = None

            # === Try OpenAI first ===
            if openai_client:
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",   # supports vision
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": "Analyze this chart and provide:\n1. Key trends\n2. Anomalies\n3. Insights\n4. Recommendations"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]}
                        ],
                        max_tokens=500
                    )
                    insights = response.choices[0].message.content
                except Exception as e:
                    st.warning(f"⚠️ OpenAI failed: {e}")

            # === Fallback to Groq ===
            if insights is None and groq_client:
                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",   # replacement for LLaVA
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": "Analyze this chart and provide:\n1. Key trends\n2. Anomalies\n3. Insights\n4. Recommendations"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]}
                        ],
                        max_tokens=500
                    )
                    insights = response.choices[0].message.content
                except Exception as e:
                    st.error(f"🚨 Groq fallback failed: {e}")

            # === Display Results ===
            if insights:
                st.success("✅ Analysis Complete")
                st.subheader("📈 Insights & Recommendations")
                st.write(insights)
