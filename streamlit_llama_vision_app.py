import streamlit as st
from groq import Groq
import base64

# === Streamlit UI Config ===
st.set_page_config(page_title="Chart Insights with Groq Vision", layout="centered")

# === Custom CSS for styling ===
st.markdown("""
    <style>
        .stApp { background-color: #fdfdfd; color: #333333; }
        h1, h2, h3, h4 { color: #1a237e; font-family: "Segoe UI", sans-serif; }
        .stButton>button {
            background: linear-gradient(90deg, #1a237e, #3949ab);
            color: white; border-radius: 8px; padding: 8px 20px;
            font-weight: bold; border: none; transition: transform 0.2s, background 0.2s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #3949ab, #1a237e); transform: scale(1.03);
        }
        .stFileUploader, .stCameraInput {
            border: 2px dashed #3949ab; border-radius: 8px; padding: 12px; background-color: #f7f8fa;
        }
    </style>
""", unsafe_allow_html=True)

# === App Header ===
st.title("📊 Chart Insights with Groq Vision")
st.subheader("Upload or capture a chart to get trends, anomalies & recommendations")

# === GROQ API Key Input ===
GROQ_API_KEY = st.text_input("Enter your GROQ_API_KEY", type="password")
if not GROQ_API_KEY:
    st.warning("Please enter your GROQ API key to enable analysis.")
    st.stop()  # Stop execution until API key is provided

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# === Image Upload / Camera Input ===
uploaded_file = st.file_uploader("📂 Upload a chart image", type=["png", "jpg", "jpeg"])
camera_file = st.camera_input("📸 Or take a photo")
image_data = uploaded_file or camera_file

if image_data:
    st.image(image_data, caption="Uploaded Chart", width=600)
    img_bytes = image_data.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    if st.button("🔍 Analyze Chart"):
        with st.spinner("Analyzing chart..."):
            try:
                # Create Groq completion request
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst interpreting charts and dashboards."},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Provide:\n1. Key trends\n2. Anomalies\n3. Business insights\n4. Next steps."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                insights = response.choices[0].message.content
                st.success("✅ Analysis Complete")
                st.subheader("📈 Insights & Recommendations")
                st.write(insights)

            except Exception as e:
                st.error(f"Error analyzing image: {e}")

else:
    st.info("Upload an image or take a photo to analyze.")
