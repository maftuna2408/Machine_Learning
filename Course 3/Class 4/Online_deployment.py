import streamlit as st
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

# Sahifa konfiguratsiyasi
st.set_page_config(
    page_title="HR Employee Attrition Predictor",
    page_icon="🔮",
    layout="centered"
)

# Modelni yuklab olish
model = load(r"C:\Users\Maftuna\Downloads\HR-Employee-Attrition.joblib")

st.markdown("<h2 style='text-align: center;'>HR Employee Attrition Prediction</h2>", unsafe_allow_html=True)
st.write("Quyidagi xodimga tegishli ma'lumotlarni kiriting va **Predict** tugmasini bosing.")

# (Bu yerda ilgari kiritilgan barcha ustunlar bo'yicha inputlar joylashadi)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
DailyRate = st.number_input("DailyRate", min_value=100, max_value=1500, value=800)
DistanceFromHome = st.number_input("DistanceFromHome", min_value=1, max_value=50, value=10)
Education = st.number_input("Education", min_value=1, max_value=5, value=3)
HourlyRate = st.number_input("HourlyRate", min_value=10, max_value=100, value=40)
JobInvolvement = st.number_input("JobInvolvement", min_value=1, max_value=4, value=3)
JobLevel = st.number_input("JobLevel", min_value=1, max_value=5, value=2)
JobSatisfaction = st.number_input("JobSatisfaction", min_value=1, max_value=4, value=3)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=50000, value=5000)
MonthlyRate = st.number_input("MonthlyRate", min_value=1000, max_value=30000, value=15000)
NumCompaniesWorked = st.number_input("NumCompaniesWorked", min_value=0, max_value=10, value=2)
PercentSalaryHike = st.number_input("PercentSalaryHike", min_value=0, max_value=30, value=15)
PerformanceRating = st.number_input("PerformanceRating", min_value=1, max_value=4, value=3)
RelationshipSatisfaction = st.number_input("RelationshipSatisfaction", min_value=1, max_value=4, value=3)
StockOptionLevel = st.number_input("StockOptionLevel", min_value=0, max_value=3, value=1)
TotalWorkingYears = st.number_input("TotalWorkingYears", min_value=0, max_value=60, value=10)
TrainingTimesLastYear = st.number_input("TrainingTimesLastYear", min_value=0, max_value=10, value=3)
WorkLifeBalance = st.number_input("WorkLifeBalance", min_value=1, max_value=4, value=3)
YearsAtCompany = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=5)
YearsInCurrentRole = st.number_input("YearsInCurrentRole", min_value=0, max_value=20, value=3)
YearsSinceLastPromotion = st.number_input("YearsSinceLastPromotion", min_value=0, max_value=20, value=2)
YearsWithCurrManager = st.number_input("YearsWithCurrManager", min_value=0, max_value=20, value=3)

BusinessTravel = st.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
EducationField = st.selectbox("EducationField", ["Human Resources", "Life Sciences", "Marketing", "Technical Degree", "Medical", "Other"])
Gender = st.selectbox("Gender", ["Female", "Male"])
JobRole = st.selectbox("JobRole", [
    "Sales Executive", "Research Scientist", "Laboratory Technician", 
    "Manufacturing Director", "Healthcare Representative", "Manager", 
    "Sales Representative", "Research Director", "Human Resources"
])
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
OverTime = st.selectbox("OverTime", ["Yes", "No"])

# Doimiy ustunlar
EmployeeCount = 1
Over18 = "Y"
StandardHours = 80
EnvironmentSatisfaction = st.number_input("EnvironmentSatisfaction", min_value=1, max_value=4, value=3)
EmployeeNumber = st.number_input("EmployeeNumber", min_value=1, max_value=999999, value=1, step=1)

# "Predict" tugmasi bosilganda
if st.button("Predict"):
    # Barcha ustunlar asosida DataFrame yaratiladi
    row = {
        "Age": Age,
        "DailyRate": DailyRate,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EmployeeCount": EmployeeCount,
        "EmployeeNumber": EmployeeNumber,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "Over18": Over18,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StandardHours": StandardHours,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "EducationField": EducationField,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus
    }
    input_df = pd.DataFrame([row])
    
    try:
        # Model orqali bashorat qilish
        prediction = model.predict(input_df)[0]
        st.write(f"**Bashorat:** Xodim attritsiya qilishi ehtimoli -> {prediction}")
    except Exception as e:
        st.error(f"Xatolik yuz berdi: {e}")
        
    # Agar haqiqiy natijalar (ground truth) mavjud bo'lsa, ularning accuracy sini ham hisoblash misoli:
    # (Agar test to'plami yoki haqiqiy qiymat kiritilgan bo'lsa, ularni o'zgaruvchi sifatida oling)
    # Misol:
    # y_true = [1]  # Foydalanuvchi tomonidan kiritilgan yoki saqlangan haqiqiy natija (bu yerda misol uchun 1)
    # acc = accuracy_score(y_true, [prediction])
    # st.write(f"Modelning aniqligi: {acc*100:.2f}%")


   
