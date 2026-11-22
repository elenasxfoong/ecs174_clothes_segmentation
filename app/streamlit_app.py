#Clothing segmentation demo (version 1)

#How to run it (from the repo folder):
#streamlit run app/streamlit_app.py

import streamlit as st #builds the web page
from PIL import Image #opens the uploaded image file
from ultralytics import YOLO #the YOLO model library


#For now this is the regular COCO-pretrained YOLO model
MODEL_PATH = "yolo26n-seg.pt"


#@st.cache_resource tells Streamlit to load the model only the first time and reuse it after that, so the app stays fast.
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

#Page layout
st.title("Clothing Detection and Segmentation")
st.write(f"Upload an outfit photo. Model in use: `{MODEL_PATH}`")

#The upload box. It returns None until the user picks a file.
uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

#Run the model when there is a photo
if uploaded_file is not None:
    #Open the file as an image. convert("RGB") makes sure PNGs with a transparent layer still work.
    image = Image.open(uploaded_file).convert("RGB")

    model = load_model()

    #Run the model, It returns a list with one result per image; we gave it one image, so we take results[0].
    results = model(image, conf=0.25)
    result = results[0]

    #result.plot() draws the masks, boxes and labels onto the photo for us.
    #It returns colors in BGR order (OpenCV's format), so [:, :, ::-1] flips them to RGB, which is what Streamlit expects.
    drawn_image = result.plot()[:, :, ::-1]

    #Show the original and the result side by side.
    left, right = st.columns(2)
    left.image(image, caption="Original", width="stretch")
    right.image(drawn_image, caption="Model output", width="stretch")


    # List what the model found
    st.subheader("Detected items")

    if len(result.boxes) == 0:
        st.write("Nothing found. Try lowering the minimum confidence.")
    else:
        rows = []
        for box in result.boxes:
            class_id = int(box.cls) #the class number, e.g. 0
            label = model.names[class_id] #turn the number into a name (read from the model, not hard-coded)
            confidence = float(box.conf) #how sure the model is, 0 to 1
            rows.append({"Item": label, "Confidence": f"{confidence:.0%}"})

        st.table(rows)
