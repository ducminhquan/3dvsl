import streamlit as st
import streamlit.components.v1 as components

import vsl

# st.title("Từ điển VSL")


options = st.multiselect(
    "Chọn các từ cần biểu diễn",
    vsl.vsldict,
)

# st.write("You selected:", options)
button = st.button("Generate 3D VSL")

if button:
    sigml_df = vsl.dictToDataFrame(options)
    sigml = vsl.convertToSigMLXML(sigml_df)

    html_file = open("vslplayer.html", "r", encoding="utf-8")
    source_code = html_file.read()

    source_code = source_code.replace("||sigml||", sigml)

    components.html(source_code, height=400)
    st.code(sigml_df, language="csv")
    st.code(sigml, language="xml")
