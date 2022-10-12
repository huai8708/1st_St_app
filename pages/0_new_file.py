import streamlit as st
st.title("this is test")

st.write(
    """
    ✨ Streamlit Elements &nbsp; [![GitHub][github_badge]][github_link] [![PyPI][pypi_badge]][pypi_link]
    =====================
    Create a draggable and resizable dashboard in Streamlit, featuring Material UI widgets,
    Monaco editor (Visual Studio Code), Nivo charts, and more!
    [github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label
    [github_link]: https://github.com/okld/streamlit-elements
    [pypi_badge]: https://badgen.net/pypi/v/streamlit-elements?icon=pypi&color=black&label
    [pypi_link]: https://pypi.org/project/streamlit-elements
    """
)


c1,c2 = st.columns(2)
b1 = c1.button("c1")
b2 = c2.button("c2")

if b1:
    st.write("this is c1")

if b2 :
    st.write("this is c2")