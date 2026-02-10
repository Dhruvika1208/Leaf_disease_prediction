import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Leaf Disease AI", layout="wide")

# ---------- UI STYLE ----------
st.markdown("""
    <style>
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .block-container { padding-top: 2rem; }
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
    return YOLO("best.pt")  # MUST be the NEW disease-trained model

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
    st.markdown("""
        <div style='background: linear-gradient(90deg, #2E8B57, #3CB371); padding: 20px; border-radius: 10px'>
            <h1 style='color: white; text-align: center;'>🌿 AI Leaf Disease Detection System</h1>
        </div><br>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:15px; border-radius:10px;'>
            <h4>👤 Logged in as: <span style='color:black'>{st.session_state.user}</span></h4>
        </div><br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("📤 Upload Leaf Image", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img = img.resize((256, 256))

        with col2:
            st.image(img, caption="Uploaded Leaf Image", width=350)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            result = model(tmp.name)[0]

        class_id = result.probs.top1
        confidence = float(result.probs.top1conf)
        label = model.names[class_id]

        # ---------- SPLIT LABEL ----------
        if "___" in label:
            crop_name, disease_name = label.split("___", 1)
        else:
            crop_name = label
            disease_name = "Healthy"

        # ---------- DISPLAY ----------
        st.markdown(f"""
            <div style='background-color:#e6ffe6; padding:20px; border-radius:12px;'>
                <h2 style='color:#006400;'>🌿 Crop: {crop_name}</h2>
            </div>
        """, unsafe_allow_html=True)

        if disease_name.lower() == "healthy":
            st.success("✅ Leaf is Healthy")
        else:
            st.error(f"🦠 Disease: {disease_name}")

        st.progress(confidence)
        st.write(f"**Confidence Score:** {confidence:.2f}")

        # ---------- DISEASE INFO ----------
        info = disease_df[disease_df[disease_col].str.lower() == disease_name.lower()][info_col].values
        if len(info) > 0:
            st.markdown(f"""
                <div style='background-color:#fff3cd; padding:15px; border-radius:10px;'>
                    <h4>🩺 Disease Information</h4>
                    <p>{info[0]}</p>
                </div>
            """, unsafe_allow_html=True)

        # ---------- SAVE HISTORY ----------
        if st.session_state.user not in history_db:
            history_db[st.session_state.user] = []

        history_db[st.session_state.user].append({
            "crop": crop_name,
            "disease": disease_name,
            "confidence": confidence
        })

        json.dump(history_db, open(history_file, "w"))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📊 Your Prediction History")

    if st.session_state.user in history_db:
        st.dataframe(pd.DataFrame(history_db[st.session_state.user]), use_container_width=True)

# ---------- SIDEBAR ----------
st.sidebar.title("Leaf Disease AI")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
else:
    page = st.sidebar.radio("Menu", ["Login", "Signup"])
    login() if page == "Login" else signup()

if st.session_state.user:
    main_app()