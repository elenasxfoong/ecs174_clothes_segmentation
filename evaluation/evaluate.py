#Grades a YOLO segmentation model on photos where we already know the right answers, then prints and saves the scores.

#How to run it (from the repo folder):
#python evaluation/evaluate.py

import os #used to create the results folder
from ultralytics import YOLO


#Which model to grade (For now regular COCO-pretrained model)
MODEL_PATH = "yolo26n-seg.pt"

#Which dataset to grade on (For now: "coco8-seg.yaml")
DATA_PATH = "coco8-seg.yaml"

#Which part of the dataset to grade on, coco8-seg only has train and val, so we use val
SPLIT = "val"

#Where to save the scores.
OUTPUT_FILE = "results/metrics/evaluation_scores.csv"


#Load the model
print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)

#model.val() runs the model on every photo in the split, compares each guess to the answer key, and calculates the scores
#plots=False skips saving extra chart images, to keep things simple.
print("Grading on dataset:", DATA_PATH, "| split:", SPLIT)
metrics = model.val(data=DATA_PATH, split=SPLIT, plots=False)

#Pull out the overall scores
box_map50 = metrics.box.map50 #metrics.box = scores for the bounding boxes, map50 = score when "correct" means at least 50% overlap
mask_map50 = metrics.seg.map50 #metrics.seg = scores for the masks (exact outlines)
mask_map50_95 = metrics.seg.map

print()
print("===== Overall scores =====")
print("Box mAP50:      ", round(box_map50, 3))
print("Mask mAP50:     ", round(mask_map50, 3))
print("Mask mAP50-95:  ", round(mask_map50_95, 3))

#Pull out the scores for each class
print()
print("===== Scores per class =====")

per_class_rows = []

for i, class_id in enumerate(metrics.ap_class_index): #metrics.ap_class_index is the list of class numbers that appear in the graded photos. For each one we look up its name and its scores.
    class_name = model.names[int(class_id)] #read the name from the model, not hard-coded

    #class_result(i) gives four numbers: precision, recall, mAP50, mAP50-95, we only keep the last two.
    box_scores = metrics.box.class_result(i)
    mask_scores = metrics.seg.class_result(i)

    class_box_map50 = box_scores[2]
    class_mask_map50 = mask_scores[2]
    class_mask_map50_95 = mask_scores[3]

    print(class_name, "| box mAP50:", round(class_box_map50, 3),
          "| mask mAP50:", round(class_mask_map50, 3),
          "| mask mAP50-95:", round(class_mask_map50_95, 3))

    per_class_rows.append([class_name, class_box_map50, class_mask_map50, class_mask_map50_95])


#Save everything to a CSV file
#The first row is the overall score ("all"), then one row per class.
os.makedirs("results/metrics", exist_ok=True)  #make the folder if it doesn't exist yet

file = open(OUTPUT_FILE, "w")
file.write(f"# model: {MODEL_PATH}, data: {DATA_PATH}, split: {SPLIT}\n")
file.write("class,box_mAP50,mask_mAP50,mask_mAP50-95\n")
file.write(f"all,{box_map50:.4f},{mask_map50:.4f},{mask_map50_95:.4f}\n")

for row in per_class_rows:
    file.write(f"{row[0]},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f}\n")

file.close()

print()
print("Saved scores to", OUTPUT_FILE)
