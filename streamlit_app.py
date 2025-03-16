import streamlit as st
import pandas as pd
import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np

# ตั้งค่าหัวข้อแอป
st.title("📌 My Enhanced Streamlit App")

# โหลดข้อมูลตัวอย่าง
DATA_URL = "https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/Automobile_insurance_fraud.csv"
df = pd.read_csv(DATA_URL)

# ฟีเจอร์ที่ใช้
selected_features = [
    "months_as_customer", "age", "policy_state", "policy_csl",
    "policy_deductable", "policy_annual_premium", "umbrella_limit", "insured_zip",
    "insured_sex", "insured_education_level", "insured_occupation", "insured_hobbies", 
    "insured_relationship", "capital-gains", "capital-loss", "incident_type", "collision_type",
    "incident_severity", "authorities_contacted", "incident_state", "incident_city", 
    "incident_hour_of_the_day", "number_of_vehicles_involved", "property_damage", 
    "bodily_injuries", "witnesses", "total_claim_amount", "injury_claim", 
    "property_claim", "vehicle_claim", "auto_make", "auto_model", "auto_year"
]

# Dropdown เลือกโมเดล
option = st.selectbox("เลือกโมเดล:", ["info1", "info2", "model1", "model2"])

if option == "info1":
    st.write("""
    โมเดลที่ใช้ในแอปนี้เป็นโมเดล Machine Learning ที่ได้รับการฝึกฝนเพื่อพยากรณ์ข้อมูลการเคลมประกัน โดยมีรายละเอียดดังนี้:

    - **Model 1 (Random Forest Classifier)**  
      เป็นอัลกอริธึมที่ใช้ต้นไม้การตัดสินใจหลายต้นมาทำงานร่วมกัน (Ensemble Learning)  
      - ทำงานได้ดีสำหรับข้อมูลที่มีความซับซ้อน  
      - ทนต่อข้อมูลที่มี Noise หรือค่าผิดปกติ  
      - ใช้สำหรับพยากรณ์ความเป็นไปได้ของการฉ้อโกงในประกัน  

    ### 📌 ข้อมูลเพิ่มเติม  
    - โมเดลได้รับการฝึกฝนด้วยชุดข้อมูลจากกรมธรรม์ประกันภัย  
    - ใช้คุณสมบัติต่างๆ เช่น อายุ, จำนวนปีที่เป็นลูกค้า, ประเภทกรมธรรม์, รายละเอียดอุบัติเหตุ เป็นต้น  
    - ปรับแต่งพารามิเตอร์เพื่อเพิ่มความแม่นยำ  

    """)

elif option == "info2":
    st.write("""
    โมเดลที่ใช้ในแอปนี้เป็นโมเดล Machine Learning ที่ได้รับการฝึกฝนเพื่อพยากรณ์ข้อมูลการเคลมประกัน โดยมีรายละเอียดดังนี้:

    - **Model 2 (Keras Neural Network)**  
      โมเดลนี้ใช้ Keras สำหรับสร้างและฝึกฝนเพื่อพยากรณ์ความเป็นไปได้ของการฉ้อโกงในประกัน  

    ### 📌 ข้อมูลเพิ่มเติม  
    - โมเดลนี้เป็นโมเดลเชิงลึก (Deep Learning) ใช้สำหรับการพยากรณ์ที่ซับซ้อน  
    - ใช้ข้อมูลจากกรมธรรม์ประกันภัย เช่น อายุ, จำนวนปีที่เป็นลูกค้า, ประเภทกรมธรรม์, รายละเอียดอุบัติเหตุ เป็นต้น  

    """)

elif option in ["model1", "model2"]:
    model_path = "model1.pkl" if option == "model1" else "model2.h5"  # ใช้ไฟล์ .h5 สำหรับ model2
    
    try:
        if option == "model1":
            # โหลด Random Forest model (.pkl)
            with open(model_path, "rb") as file:
                model = pickle.load(file)
        else:
            # โหลด Keras model (.h5)
            model = load_model(model_path)  # ใช้ load_model จาก Keras

        st.write("📥 **กรอกค่าฟีเจอร์สำหรับพยากรณ์:**")

        if option == "model2":
            # If model2 is selected, let the user upload an image
            uploaded_image = st.file_uploader("อัพโหลดรูปภาพสำหรับพยากรณ์:", type=["jpg", "jpeg", "png"])
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image.", use_column_width=True)

                # Preprocess the image before feeding it into the model
                image = image.resize((50, 50))  # Resize the image to 50x50 (expected input size)
                image = np.array(image) / 255.0  # Normalize the image

                # Check if the image has 3 channels (RGB)
                if len(image.shape) == 2:  # If grayscale image, convert to RGB
                    image = np.stack([image] * 3, axis=-1)

                # Ensure the image has 3 channels and has shape (50, 50, 3)
                if image.shape[2] != 3:
                    st.error("🚨 รูปภาพต้องมี 3 ช่องสี (RGB).")
                else:
                    # Expand dimensions to match the input shape of the model
                    image = np.expand_dims(image, axis=0)  # Shape (1, 50, 50, 3)

                    # Make prediction with the model
                    prediction = model.predict(image)

                    st.success(f"📊 ค่าพยากรณ์จาก Model2: {prediction[0]}")

        else:
            # For model1, handle inputs as before
            user_inputs = {}
            user_inputs["months_as_customer"] = st.selectbox("Months as Customer", list(range(1, 121)))
            user_inputs["age"] = st.selectbox("Age", list(range(18, 100)))
            user_inputs["policy_state"] = st.selectbox("Policy State", ["CA", "TX", "NY"])
            user_inputs["policy_csl"] = st.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
            user_inputs["policy_deductable"] = st.selectbox("Policy Deductible", [0, 500, 1000, 2000])
            user_inputs["policy_annual_premium"] = st.selectbox("Policy Annual Premium", [500.00, 1000.00, 1500.00, 2000.00])
            user_inputs["umbrella_limit"] = st.selectbox("Umbrella Limit", [0, 100000, 200000, 300000])
            user_inputs["insured_zip"] = st.selectbox("Insured ZIP Code", ["90001", "10001", "77001"])
            user_inputs["insured_sex"] = st.selectbox("Insured Sex", ["Male", "Female"])
            user_inputs["insured_education_level"] = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
            user_inputs["insured_occupation"] = st.selectbox("Insured Occupation", ["Engineer", "Doctor", "Teacher", "Lawyer"])
            user_inputs["insured_hobbies"] = st.selectbox("Insured Hobbies", ["Golf", "Reading", "Traveling", "Cooking"])
            user_inputs["insured_relationship"] = st.selectbox("Relationship Status", ["Single", "Married"])
            user_inputs["capital-gains"] = st.selectbox("Capital Gains", [0, 5000, 10000, 20000])
            user_inputs["capital-loss"] = st.selectbox("Capital Loss", [0, 1000, 2000, 5000])
            user_inputs["incident_type"] = st.selectbox("Incident Type", ["Single Vehicle", "Multi-Vehicle"])
            user_inputs["collision_type"] = st.selectbox("Collision Type", ["Front Collision", "Rear Collision", "Side Collision"])
            user_inputs["incident_severity"] = st.selectbox("Incident Severity", ["Minor Injury", "Major Injury", "Severe Injury"])
            user_inputs["authorities_contacted"] = st.selectbox("Authorities Contacted", ["Police", "Fire Department", "Ambulance", "None"])
            user_inputs["incident_state"] = st.selectbox("Incident State", ["CA", "TX", "NY"])
            user_inputs["incident_city"] = st.selectbox("Incident City", ["Los Angeles", "Houston", "New York"])
            user_inputs["incident_hour_of_the_day"] = st.selectbox("Incident Hour", list(range(0, 24)))
            user_inputs["number_of_vehicles_involved"] = st.selectbox("Number of Vehicles Involved", [1, 2, 3, 4])
            user_inputs["property_damage"] = st.selectbox("Property Damage", ["YES", "NO"])
            user_inputs["bodily_injuries"] = st.selectbox("Bodily Injuries", [0, 1, 2, 3])
            user_inputs["witnesses"] = st.selectbox("Witnesses", [0, 1, 2, 3, 4])
            user_inputs["total_claim_amount"] = st.selectbox("Total Claim Amount", [1000, 5000, 10000, 20000])
            user_inputs["injury_claim"] = st.selectbox("Injury Claim", [0, 2000, 5000, 10000])
            user_inputs["property_claim"] = st.selectbox("Property Claim", [0, 1000, 3000, 5000])
            user_inputs["vehicle_claim"] = st.selectbox("Vehicle Claim", [0, 2000, 4000, 6000])
            user_inputs["auto_make"] = st.selectbox("Auto Make", ["Toyota", "Honda", "Ford", "BMW"])
            user_inputs["auto_model"] = st.selectbox("Auto Model", ["Camry", "Civic", "F-150", "X5"])
            user_inputs["auto_year"] = st.selectbox("Auto Year", list(range(1980, 2025)))

            # กดปุ่มแล้วทำพยากรณ์
            if st.button("🔮 พยากรณ์ผลลัพธ์"):
                try:
                    input_df = pd.DataFrame([user_inputs])
                    label_encoder = LabelEncoder()
                    categorical_features = input_df.select_dtypes(include=['object']).columns

                    for col in categorical_features:
                        input_df[col] = label_encoder.fit_transform(input_df[col])

                    input_values = input_df.values.reshape(1, -1)

                    if option == "model1":
                        prediction = model.predict(input_values)
                    else:  # Model2 is a Keras model
                        prediction = model.predict(input_values)

                    st.success(f"📊 ค่าพยากรณ์จาก {option}: {prediction[0]}")
                except ValueError:
                    st.error("🚨 กรุณากรอกค่าที่ถูกต้องในแต่ละฟีเจอร์")
    except FileNotFoundError:
        st.error(f"🚨 ไม่พบไฟล์โมเดล {model_path}")
    except Exception as e:
        st.error(f"🚨 เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
