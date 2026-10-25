import streamlit as st 

st.image("calc.png",width=500)
st.title("Calculator")

st.write("This is a Calculator")

num1 = st.number_input("Enter first number:",max_value=100,min_value=1,value=None,placeholder="Enter integer value")
num2 = st.number_input("Enter second number:",max_value=100,min_value=1,value=None,placeholder="Enter integer value")


if st.button("ADD"):
    result=num1+num2
elif st.button("SUB"):
    result=num1-num2
elif st.button("MUL"):
    result=num1*num2
elif st.button("DIV"):
    result=num1/num2

    st.write("Result="+str(result))