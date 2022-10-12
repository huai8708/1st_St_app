import streamlit as st
st.title("this is test")

c1,c2 = st.columns(2)
b1 = c1.button("c1")
b2 = c2.button("c2")

if b1:
    st.write("this is c1")

if b2 :
    st.write("this is c2")