import streamlit as st
import pandas as pd
import joblib
import time

# =======================
# Load Dataset & Model
# =======================
data = pd.read_csv("merged_real_estate_data.csv")
model = joblib.load("real_estate_price_model.pkl")

# Clean columns
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Detect price column
price_col = None
for col in data.columns:
    if "price" in col.lower():
        price_col = col
        break

if price_col is None:
    st.error("❌ No price column found in the dataset.")
    st.stop()

# =======================
# Page Config
# =======================
st.set_page_config(page_title="🏡 Real Estate Price Predictor", layout="centered")

# 🎨 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    * { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #6a11cb, #2575fc, #ff6a00, #ffcc70);
        background-size: 300% 300%;
        animation: gradientMove 12s ease infinite;
        color: #fff;
    }

    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    h1 {
        text-align: center;
        font-size: 2.8em !important;
        font-weight: 700 !important;
        color: #fff;
        text-shadow: 0px 3px 12px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.12);
        padding: 30px;
        border-radius: 18px;
        max-width: 750px;
        margin: 20px auto;
        box-shadow: 0 8px 40px rgba(0,0,0,0.3);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
    }

    label { font-weight: 600 !important; color: #fff !important; }

    /* ✅ Black text for Area & Locality */
    input[type="number"], input[type="text"] {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        padding: 6px 10px !important;
    }

    .floating-btn {
        background: linear-gradient(90deg, #ff6a00, #ffcc70);
        color: #000 !important;
        font-weight: 700;
        border-radius: 50px;
        padding: 14px 40px;
        font-size: 18px;
        width: 100%;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        border: none;
        margin-top: 15px;
    }

    .floating-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }

    .price-card {
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        color: #fff;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-top: 40px;
        text-shadow: 0px 2px 8px rgba(0,0,0,0.4);
        animation: fadeIn 0.9s ease-in-out;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: scale(0.8);}
        to {opacity: 1; transform: scale(1);}
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# PRICE PREDICTOR
# =======================
st.title("🏠 AI Real Estate Price Predictor")

# ✅ Directly put inputs without creating an empty glass-card first
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("📐 Area (in sq. ft)", min_value=100, max_value=10000, step=10)
        bedrooms = st.selectbox("🛏️ Bedrooms (BHK)", [1, 2, 3, 4, 5])
    with col2:
        locality = st.text_input("📍 Locality")
        city = st.text_input("🏙️ City")

    predict_clicked = st.button("🚀 Predict Price", key="predict", help="Estimate property price")

    st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'locality': locality,
        'city': city,
    }])

    try:
        progress = st.progress(0)
        status = st.empty()
        dots = ["", ".", "..", "..."]
        for i in range(100):
            progress.progress(i + 1)
            status.markdown(f"<h4 style='color:white;'>🔍 Analyzing property data{dots[i % 4]}</h4>", unsafe_allow_html=True)
            time.sleep(0.015)

        progress.empty()
        status.empty()

        final_price = model.predict(input_data)[0]

        placeholder = st.empty()
        steps = 25
        for i in range(1, steps + 1):
            animated_value = final_price * (i / steps)
            placeholder.markdown(
                f"<div class='price-card'>💰 Estimated Price: ₹ {animated_value:,.0f}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.03)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# =======================
# FOOTER
# =======================
st.markdown("""
<hr style="border-top: 1px solid rgba(255,255,255,0.2);">
<center>
    <span style="color:#fff; font-size:14px;">© 2025 NextGen Real Estate AI · Powered by Streamlit</span>
</center>
""", unsafe_allow_html=True)
