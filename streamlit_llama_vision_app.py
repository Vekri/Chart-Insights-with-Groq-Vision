import os
import base64
import io
import mimetypes
from typing import Optional, Tuple
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
# Exception classes (SDK v1)
from openai import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    BadRequestError,
    APIError,
)

# === Streamlit UI Config ===
st.set_page_config(page_title="Chart Insights with LLM", layout="centered")

st.markdown("""
    <style>
        .stApp { background-color: #fdfdfd; color: #333333; }
        h1, h2, h3, h4 { color: #1a237e; font-family: "Segoe UI", sans-serif; }
        .stButton>button {
            background: linear-gradient(90deg, #1a237e, #3949ab);
            color: #ffffff; border-radius: 8px; padding: 8px 20px;
            font-weight: bold; border: none; transition: transform 0.2s, background 0.2s;
        }
        .stButton>button:hover { background: linear-gradient(90deg, #3949ab, #1a237e); transform: scale(1.03); }
        .stFileUploader, .stCameraInput {
            border: 2px dashed #3949ab; border-radius: 8px; padding: 12px; background-color: #f7f8fa;
        }
        .tip { font-size: 0.9rem; color: #455a64; }
    </style>
""", unsafe_allow_html=True)

# === Helpers ===
def get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets; fallback to env var for local quick runs
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY")

def render_setup_help():
    with st.expander("🛠 How to enable the API (ChatGPT Go users too)"):
        st.markdown("""
1. **Create an API key** (developer platform)
   - Go to **platform.openai.com**
   - Dashboard → **API keys** → **Create new secret key**
2. **Add a payment method** (API is pay‑as‑you‑go)
   - Dashboard → **Billing** → **Add payment method**
3. **Add your key to the app**
   - EITHER set environment var locally:
     ```bash
     export OPENAI_API_KEY="sk-...yourkey..."
     ```
   - OR create `.streamlit/secrets.toml`:
     ```toml
     OPENAI_API_KEY = "sk-...yourkey..."
     ```
        """)
        st.info("ChatGPT Go is for the ChatGPT app only. The Streamlit app uses the **OpenAI API**; you still need an API key and billing on the developer platform.")

def guess_mime(filename: str, default="image/png") -> str:
    m, _ = mimetypes.guess_type(filename or "")
    return m or default

def image_to_data_url(file, default_mime="image/png") -> Tuple[str, int]:
    content = file.getvalue()
    b64 = base64.b64encode(content).decode("utf-8")
    mime = guess_mime(getattr(file, "name", ""), default_mime)
    return f"data:{mime};base64,{b64}", len(content)

def truncate_table_for_prompt(df: pd.DataFrame, max_rows: int = 50, max_chars: int = 8000) -> str:
    """
    Keep the prompt safe in size: limit rows and characters.
    """
    # Keep numeric columns primarily for analysis
    numeric_df = df.select_dtypes(include="number")
    small = (numeric_df if not numeric_df.empty else df).head(max_rows)
    csv_text = small.to_csv(index=False)
    if len(csv_text) > max_chars:
        csv_text = csv_text[:max_chars] + "\n... [truncated]"
    return csv_text

def plot_quick(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None):
    plt.figure()
    if x_col and x_col in df.columns:
        df.plot(x=x_col, y=y_col, kind="line", legend=False)  # default colors only
    else:
        df[y_col].plot(kind="line", legend=False)
    plt.title(f"{y_col} over {x_col or 'index'}")
    plt.xlabel(x_col or "index")
    plt.ylabel(y_col)
    st.pyplot(plt.gcf())

def analyze_image_with_llm(client: OpenAI, data_url: str, model: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,  # e.g., "gpt-4o-mini"
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text":
                 "You are a senior data analyst. Analyze this chart and provide:\n"
                 "1) Key trends\n2) Anomalies/outliers\n3) Business insights\n4) Recommended next steps "
                 "with metrics that could be tracked next and any quick checks to validate the insights."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def analyze_table_with_llm(client: OpenAI, table_csv: str, model: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text":
                 "You are a senior data analyst. I will provide a small sample of a dataset (CSV text). "
                 "Infer the likely chart patterns and provide:\n"
                 "1) Key trends\n2) Anomalies/outliers\n3) Business insights\n4) Recommended next steps "
                 "and which visualizations would best reveal them.\n\nHere is the sample:\n\n" + table_csv}
            ]
        }],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def friendly_error(e: Exception) -> str:
    if isinstance(e, AuthenticationError):
        return "Authentication failed. Check that your **API key** is valid."
    if isinstance(e, RateLimitError):
        return "Rate limit or quota reached. Check **billing/quota** on the platform."
    if isinstance(e, APIConnectionError):
        return "Network issue connecting to OpenAI. Please retry or check your internet."
    if isinstance(e, BadRequestError):
        return "Bad request (likely input format/model mismatch)."
    if isinstance(e, APIError):
        return f"OpenAI API error: {getattr(e, 'message', str(e))}"
    return f"{type(e).__name__}: {str(e)}"

# === App Header ===
st.title("📊 Chart Insights with LLM")
st.subheader("Upload a chart image or raw data to get trends, anomalies & recommendations")

# === Sidebar: Settings ===
st.sidebar.header("⚙️ Settings")
model = st.sidebar.selectbox(
    "Model",
    options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"],  # include options you have access to
    index=0
)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 128, 2000, 700, 64)

# === API client init with checks ===
api_key = get_api_key()
client = None
api_ready = True
if not api_key:
    st.warning("🚨 No API key detected. Add it via `.streamlit/secrets.toml` or `OPENAI_API_KEY`.")
    render_setup_help()
    api_ready = False
else:
    try:
        client = OpenAI(api_key=api_key)
        # light ping (no request — rely on first call error handling)
    except Exception as e:
        st.error("Could not initialize OpenAI client. " + friendly_error(e))
        render_setup_help()
        api_ready = False

# === Tabs for inputs ===
tab_img, tab_data = st.tabs(["🖼️ Image (Chart)", "📑 Data (CSV/Excel)"])

with tab_img:
    uploaded_file = st.file_uploader("📂 Upload a chart image", type=["png", "jpg", "jpeg"])
    camera_file = st.camera_input("📸 Or take a photo (rear camera preferred)", key="rear_camera_input")
    image_data = uploaded_file or camera_file

    if image_data:
        st.image(image_data, caption="Uploaded Chart", width=600)
        if st.button("🔍 Analyze Image", use_container_width=True):
            if not api_ready:
                st.stop()
            with st.spinner("Analyzing image..."):
                try:
                    data_url, size_bytes = image_to_data_url(image_data)
                    if size_bytes > 14 * 1024 * 1024:
                        st.error("Image is too large. Please upload an image under ~14 MB.")
                    else:
                        insights = analyze_image_with_llm(client, data_url, model, temperature, max_tokens)
                        st.success("✅ Analysis Complete")
                        st.subheader("📈 Insights & Recommendations")
                        st.write(insights)
                except Exception as e:
                    st.error("Error analyzing image: " + friendly_error(e))
                    if isinstance(e, (AuthenticationError, RateLimitError)):
                        render_setup_help()

with tab_data:
    data_file = st.file_uploader("📥 Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    show_quick_plot = st.checkbox("Show a quick line plot for a numeric column", value=False)

    if data_file:
        try:
            if data_file.name.lower().endswith(".csv"):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)

            st.write("**Preview (first 15 rows):**")
            st.dataframe(df.head(15), use_container_width=True)

            # Optional quick plot for a selected numeric column
            if show_quick_plot:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    y_col = st.selectbox("Choose numeric column to plot", options=num_cols, index=0)
                    x_options = ["(index)"] + df.columns.tolist()
                    x_choice = st.selectbox("X axis", options=x_options, index=0)
                    x_col = None if x_choice == "(index)" else x_choice
                    plot_quick(df, y_col=y_col, x_col=x_col)
                else:
                    st.info("No numeric columns detected for plotting.")

            if st.button("🔎 Analyze Data", use_container_width=True):
                if not api_ready:
                    st.stop()
                with st.spinner("Generating insights from data sample..."):
                    try:
                        sample_csv = truncate_table_for_prompt(df, max_rows=80, max_chars=9000)
                        insights = analyze_table_with_llm(client, sample_csv, model, temperature, max_tokens)
                        st.success("✅ Analysis Complete")
                        st.subheader("📊 Data Insights & Recommendations")
                        st.write(insights)
                    except Exception as e:
                        st.error("Error analyzing data: " + friendly_error(e))
                        if isinstance(e, (AuthenticationError, RateLimitError)):
                            render_setup_help()
        except Exception as e:
            st.error(f"Failed to read the file: {e}")

# === Footer tip ===
st.markdown(
    "<p class='tip'>Tip: For best results with images, upload a clear chart (no glare, high contrast). "
    "For data, include key metrics and a time/index column where possible.</p>",
    unsafe_allow_html=True
)

