import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ตั้งค่าหัวข้อแอป
st.title("📌 My Enhanced Streamlit App")

# Dropdown เลือกข้อมูล
option = st.selectbox("เลือกหมวดหมู่ที่ต้องการดูข้อมูล:", ["info1", "info2", "model1", "model2"])

# แสดงผลตามตัวเลือก
st.subheader(f"🔍 ข้อมูลที่เลือก: {option}")

data = None
if option in ["info1", "info2"]:
    data = pd.DataFrame({
        "ประเภท": ["A", "B", "C", "D"],
        "ค่าเฉลี่ย": np.random.rand(4) * 100
    })
    st.bar_chart(data.set_index("ประเภท"))

elif option in ["model1", "model2"]:
    # โหลดโมเดลที่บันทึกไว้ (ใส่ path ที่ถูกต้อง)
    model_path = "model1.pkl" if option == "model1" else "model2.pkl"
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        
        # สร้างข้อมูลตัวอย่าง
        sample_data = np.random.rand(1, 5)  # สมมติว่ามี 5 ฟีเจอร์
        prediction = model.predict(sample_data)

        st.write(f"📊 ค่าพยากรณ์จาก {option}: {prediction[0]}")
    except FileNotFoundError:
        st.error(f"🚨 ไม่พบไฟล์โมเดล {model_path}")

# ปุ่มสุ่มข้อมูลใหม่
if st.button("🔄 สุ่มข้อมูลใหม่"):
    st.experimental_rerun()
