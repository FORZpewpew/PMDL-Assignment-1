import streamlit as st
import requests
from PIL import Image
import io

st.title("Grayscale Cats Colorization App")
st.text("Cats colorization because model was trained on pictures of cats")
uploaded_file = st.file_uploader("Upload a grayscale cat image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to a format that can be sent to the API
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    if st.button("Make Prediction"):
        # Send the image to the FastAPI API
        files = {'file': img_buffer}
        response = requests.post("http://api:8000/predict/", files=files)

        if response.status_code == 200:
            # Convert the response content to an image and display it
            output_image = Image.open(io.BytesIO(response.content))
            st.image(output_image, caption='Colorized Image', use_column_width=True)
        else:
            st.write("Error in prediction")
