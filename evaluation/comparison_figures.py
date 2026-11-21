#Runs the models on the SAME outfit photos and saves the results as pictures, so we can see the difference
#"Before" = the regular COCO-pretrained YOLO model (doesn't know clothing)
#"After"  = our model fine-tuned on DeepFashion2

#QUALITATIVE comparison (pictures, not scores). The two models use different class lists, so we never compare their mAP numbers.

#How to run it (from the repo folder):
#1. Put your outfit photos in the INPUT_FOLDER below.
#2. python evaluation/comparison_figures.py

import os #for working with folders and file names
import matplotlib.pyplot as plt #for putting two pictures side by side
from ultralytics import YOLO #the YOLO model library


# The "before" model: the regular COCO-pretrained YOLO model
BEFORE_MODEL_PATH = "yolo26n-seg.pt"

# The "after" model: ours fine-tuned model
AFTER_MODEL_PATH = "models/best.pt"

#folder of outfit photos to compare (not train)
INPUT_FOLDER = "data/comparison_photos"

#folder for finished pictures
OUTPUT_FOLDER = "results/figures/comparison"

#Only show guesses the model is at least this sure about
CONFIDENCE = 0.25

#Load the models
print("Loading the BEFORE model:", BEFORE_MODEL_PATH)
before_model = YOLO(BEFORE_MODEL_PATH)

#Check if our model file is there yet.
if os.path.exists(AFTER_MODEL_PATH):
    print("Loading the AFTER model:", AFTER_MODEL_PATH)
    after_model = YOLO(AFTER_MODEL_PATH)
else:
    print("No AFTER model found at", AFTER_MODEL_PATH, "- saving BEFORE pictures only.")
    after_model = None

#Find the photos

#Make the output folder if it doesn't exist yet.
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Go through every file in the input folder and keep only the photos.
photo_names = []
for file_name in sorted(os.listdir(INPUT_FOLDER)):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        photo_names.append(file_name)

print("Found", len(photo_names), "photos in", INPUT_FOLDER)

#Run the models on each photo and save the pictures
for file_name in photo_names:
    photo_path = os.path.join(INPUT_FOLDER, file_name)
    print("Working on:", file_name)

    
    before_result = before_model(photo_path, conf=CONFIDENCE)[0] #Run the BEFORE model and draw its masks, boxes and labels.
    before_picture = before_result.plot()[:, :, ::-1] #[:, :, ::-1] flips the colors from BGR to RGB

    #Only the BEFORE model, one picture
    if after_model is None:
        figure, axis = plt.subplots(1, 1, figsize=(6, 8))
        axis.imshow(before_picture)
        axis.set_title("Before: COCO-pretrained YOLO")
        axis.axis("off") #hide the x/y number lines around the picture

    #Both models, two pictures side by side
    else:
        after_result = after_model(photo_path, conf=CONFIDENCE)[0]
        after_picture = after_result.plot()[:, :, ::-1]

        #1 row, 2 columns: left = before, right = after
        figure, axes = plt.subplots(1, 2, figsize=(12, 8))

        axes[0].imshow(before_picture)
        axes[0].set_title("Before: COCO-pretrained YOLO")
        axes[0].axis("off")

        axes[1].imshow(after_picture)
        axes[1].set_title("After: fine-tuned on DeepFashion2")
        axes[1].axis("off")

    #Save the picture. "photo.jpg" becomes "photo_comparison.png".
    name_without_extension = os.path.splitext(file_name)[0]
    output_path = os.path.join(OUTPUT_FOLDER, name_without_extension + "_comparison.png")

    figure.savefig(output_path, dpi=150, bbox_inches="tight")#dpi = image sharpness. bbox_inches="tight" trims the empty white space around the pictures.
    plt.close(figure) #free the memory before the next photo

    print("  Saved:", output_path)

print()
print("Done! Pictures are in", OUTPUT_FOLDER)
