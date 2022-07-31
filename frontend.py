

import streamlit as st
import function as fn
from function import *


st.title("ATTENDANCE SYSTEM USING FACE RECOGNITION")
st.subheader("This is a attendance system for recording attendance through an image of the class")

st.markdown("**************")
st.title("Record Attendance")
imagefile = st.file_uploader("select an image of the class", type=(["jpg", "png"]))
if imagefile is not None:
    from pathlib import Path
    path = Path(imagefile.name)
else:
    path = None
if st.button("Generate Attendance"):
    success, count = fn.attendance(path)
    if success:
        st.success("Identified "+str(count)+" students")
        st.success("Image saved")
    #st.image(imagefile, use_column_width=True, clamp=True)

