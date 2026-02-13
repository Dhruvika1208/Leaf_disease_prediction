import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="AI Crop Health Scanner", layout="wide")

# 🌌 ---------- FUTURISTIC UI STYLE ----------
st.markdown("""
<style>

/* 🌌 Dark AI Background */
body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}

/* ✨ Glass Effect Cards */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
    text-align:center;
    margin-bottom:20px;
}

/* 🌟 Neon Button */
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 20px;
    height: 3em;
    font-size: 16px;
    box-shadow: 0 0 15px #00c6ff;
}

/* 📤 Upload Glow */
.stFileUploader {
    border: 2px dashed #00c6ff;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
users_file = os.path.join(BASE_DIR, "users.json")
history_file = os.path.join(BASE_DIR, "history.json")
disease_file = os.path.join(BASE_DIR, "disease_info.csv")

# ---------- INIT FILES ----------
if not os.path.exists(users_file):
    json.dump({}, open(users_file, "w"))

if not os.path.exists(history_file):
    json.dump({}, open(history_file, "w"))

users_db = json.load(open(users_file))
history_db = json.load(open(history_file))

# ---------- LOAD DISEASE INFO ----------
disease_df = pd.read_csv(disease_file)
disease_df.columns = disease_df.columns.str.lower()

disease_col = [c for c in disease_df.columns if "disease" in c][0]
info_col = [c for c in disease_df.columns if "info" in c or "description" in c][0]

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------- SESSION ----------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------- SIGNUP ----------
def signup():
    st.title("📝 Signup")
    name = st.text_input("Full Name")
    email = st.text_input("Email").strip().lower()
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):
        if email in users_db:
            st.error("User already exists")
        else:
            users_db[email] = {"name": name, "password": password}
            json.dump(users_db, open(users_file, "w"))
            st.success("Account created! Please login.")

# ---------- LOGIN ----------
def login():
    st.title("🔐 Login")
    email = st.text_input("Email").strip().lower()
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email in users_db and users_db[email]["password"] == password:
            st.session_state.user = users_db[email]["name"]
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

# ---------- MAIN APP ----------
def main_app():

    # 🌌 FUTURISTIC HEADER
    st.markdown("""
    <div class="glass">
        <h1>🧠 AI Crop Health Scanner</h1>
        <p>Deep Learning Powered Disease Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass">
        <h3>👤 Logged in as: {st.session_state.user}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📤 Upload Leaf Image for AI Scan")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img = img.resize((256, 256))

        with col2:
            st.image(img, caption="Scanned Leaf Image", width=350)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            result = model(tmp.name)[0]

        class_id = result.probs.top1
        confidence = float(result.probs.top1conf)
        label = model.names[class_id]

        clean_label = label.replace("_", " ").strip()

        if "___" in clean_label:
            crop_name, disease_name = clean_label.split("___", 1)
        else:
            parts = clean_label.split(" ")
            crop_name = parts[0]
            disease_name = " ".join(parts[1:])

        # 🌿 PLANT IDENTIFIED PANEL
        st.markdown(f"""
        <div class="glass">
            <h2>🌿 Plant Identified</h2>
            <h1>{crop_name}</h1>
        </div>
        """, unsafe_allow_html=True)

        # 🦠 DISEASE STATUS PANEL
        if "healthy" in disease_name.lower():
            status = "✅ HEALTHY"
        else:
            status = disease_name

        st.markdown(f"""
        <div class="glass">
            <h2>🦠 Disease Status</h2>
            <h1>{status}</h1>
        </div>
        """, unsafe_allow_html=True)

        # ⚡ AI CONFIDENCE
        st.markdown("### ⚡ AI Confidence Level")
        st.progress(confidence)

        # 🩺 DISEASE INFO
        info = disease_df[disease_df[disease_col].str.lower() == disease_name.lower()][info_col].values
        if len(info) > 0:
            st.markdown(f"""
            <div class="glass">
                <h3>🩺 Disease Information</h3>
                <p>{info[0]}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📊 Prediction History")
    if st.session_state.user in history_db:
        st.dataframe(pd.DataFrame(history_db[st.session_state.user]), use_container_width=True)

# ---------- SIDEBAR ----------
st.sidebar.title("🌌 AI Scanner")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
else:
    page = st.sidebar.radio("Menu", ["Login", "Signup"])
    login() if page == "Login" else signup()

if st.session_state.user:
    main_app()