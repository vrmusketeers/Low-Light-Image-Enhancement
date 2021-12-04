import streamlit as st
import requests
import base64
import PIL.Image as Image
import matplotlib.pyplot as plt
import io
import time

# defining the api-endpoint
API_ENDPOINT = "https://zerodcnpred-yjmgrhmjwq-uc.a.run.app/predict"
st.title('Low-Light Image Enhancement')
st.subheader('Upload a png/ jpeg file to improve the image resolution')

def upload_improve_resolution():
    uploaded_file = st.file_uploader("Choose a file")
    
    col1,col2 = st.columns(2) 
    
    if uploaded_file is not None:
        # To read file as bytes:
        image = Image.open(uploaded_file)
        st.write("")
        
        with col1:    
            image.show()
            st.image(image, caption='Low resolution Input')
            st.write("")

        encoded_content = base64.b64encode(uploaded_file.getvalue())

        data = {'instances': encoded_content}
        try:
            r = requests.post(url=API_ENDPOINT, data=data)
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)

        with col2:
            if r:
                bts = r.json()['predictions']
                image = Image.open(io.BytesIO(base64.urlsafe_b64decode(bts)))
                plt.imshow(image)
                st.image(image, caption='High resolution Output')
            else:
                st.write("Error!! Please check the resolution of input image")



if __name__ == '__main__':
    upload_improve_resolution()
    # Select a file
