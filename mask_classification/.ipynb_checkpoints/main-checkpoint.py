import streamlit as st

st.set_page_config(page_title="Hello",page_icon= "👋")

st.write("# Welcome to CV-12 Mask Classification Demo Page! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This page shows our first project demo about mask classification with age and gender!
    **👈 Select a demo from the sidebar** to see our models works!
    
    ### Want to see our project history?
    - Check out [Our project repo](https://github.com/boostcampaitech4cv3/level1_imageclassification_cv-level1-cv-20)
    
    ### Teammates
    - JaeYoung Shin  
        [Github](https://github.com/LimePencil)
    - SangJun Yoon  
        [Github](https://github.com/SangJunni)
    - Jiyong Jeon  
        [Github](https://github.com/Jiyong-Jeon)
    - YoungSeob Lee  
        [Github](https://github.com/0seob)
    - WonJoon Seo  
        [Github](https://github.com/won-joon)
"""
)