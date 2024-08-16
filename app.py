import streamlit as st
import time

pages = {
    "3D VSL": [
        
        st.Page("3dvsl.py", title="Chuyển đổi câu thành 3D VSL"),
        st.Page("dictionary.py", title="Từ điển VSL"),
    ],
    "Resources": [
        # st.Page("learn.py", title="Learn about us"),
        # st.Page("trial.py", title="Try it out"),
    ],
}

pg = st.navigation(pages)
pg.run()
