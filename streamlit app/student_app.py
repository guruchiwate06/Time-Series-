import streamlit as st 

#Title
st.title("Student info app")

#Image 
st.image("https://www.google.com/imgres?q=student&imgurl=https%3A%2F%2Fst2.depositphotos.com%2F1003369%2F7653%2Fi%2F450%2Fdepositphotos_76535327-stock-photo-happy-african-american-college-student.jpg&imgrefurl=https%3A%2F%2Fdepositphotos.com%2Fphotos%2Fafrican-student-png.html&docid=_V1ylu0ulyr0zM&tbnid=6HwhReEUBGYtSM&vet=12ahUKEwj59Lam-YOXAxVKjuEIHSInDlcQnPAOegQIMhAA..i&w=429&h=600&hcb=2&ved=2ahUKEwj59Lam-YOXAxVKjuEIHSInDlcQnPAOegQIMhAA",
         caption="Learning with tech"
         )
#Text
st.write("Enter the details below.")

#Text input
name = st.text_input("Enter your name:")
#Number input 
age = st.number_input("Enter your age:",min_value=1,max_value=100,value=None)

gender = st.radio(
    "Select you gender",
    ["Male","Female","No Other"]
)

#Checkboxes
st.write("SKILLS:")
st.checkbox("Python")
st.checkbox("Java")
st.checkbox("C++")
st.checkbox("Ruby")

#Radio
gender = st.radio(
    "Select Course",
    ["Btech","BBA","MBA"]
)

#Slider
rating = st.slider("Rate your programming skills",1,10,5)

#File uploder

uploded_file = st.file_uploader(
    "Uplode a CV file",
    type=["docx","pdf"]

)