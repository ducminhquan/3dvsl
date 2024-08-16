import streamlit as st
import streamlit.components.v1 as components

import vsl

# st.title("Chuyển đổi câu thành 3D VSL")

common_sentences = [
    "Bạn có bị đau bụng không?",
    "Nhớ uống thuốc đúng giờ!",
    "Nhớ chăm chỉ tập thể dục",
    "Thứ hai tuần sau bạn đến nhận kết quả",
    "Bạn bị đau ở đâu, hãy chỉ cho tôi biết",
]

# with st.form(key="form"):
predefined = st.selectbox("Các mẫu câu thường dùng", common_sentences, index=None)
st.write("hoặc")
original = st.text_input("Nhập vào câu cần biểu diễn", "", disabled=predefined is not None)
        
    # submit = st.form_submit_button("Submit")
    
    # if submit:
    #     st.write("You submitted the form")
    #     # st.write(original)
    #     # st.stop()


button = st.button("Generate 3D VSL")
if button:
    original = predefined if predefined is not None else original
    sigml_df = vsl.convertToSiGML(original)
    tokenize = sigml_df[0]
    vsl_string = sigml_df[1]
    sigml = vsl.convertToSigMLXML(sigml_df[2])

    # read html file vslplayer.html and render it
    html_file = open("vslplayer.html", "r", encoding="utf-8")
    source_code = html_file.read()

    # find and replace ||url|| with original
    source_code = source_code.replace("||sigml||", sigml)

    components.html(source_code, height=400)
    st.code(tokenize)
    st.code(vsl_string)
    st.code(sigml_df[2], language="csv")
    st.code(sigml, language="xml")
