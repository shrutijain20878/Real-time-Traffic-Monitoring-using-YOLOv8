import streamlit as st
import tempfile
from src.pipeline import process_video

st.title("🚦 Traffic Analytics System")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file:
    st.video(uploaded_file)

    if st.button("Run Analysis"):

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name

        output_path = "output.mp4"

        result, counts, overspeed = process_video(input_path, output_path)

        st.success("Processing Done!")
        st.video(result)

        with open(result, "rb") as file:
            st.download_button("Download Output", file, "output.mp4")


        st.subheader(" Counts")
        for k, v in counts.items():
            st.metric(k, v)
        st.subheader(" Overspeed Violations")
        st.metric("Total Violations", overspeed)