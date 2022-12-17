"""
AUTHOR: "Ayush Dulal, Prashant G C, Prashant Purie, Raj Dhaubanjar, Sumit Shrestha"
FILENAME: main.py
SPECIFICATION: "Classfication and Detection of Animals in an image"
FOR: "CS 5364 Information Retrieval Section 001"
"""

# Importing Libraries
import cv2
import numpy as np
import torch
import base64
import uvicorn
import argparse
import random
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from typing import List

# Initialize the FASTAPI application
app = FastAPI()

# Choosing the template directory for view: htmlFile
templates = Jinja2Templates(directory="templates")

# Setting up our custom model to model
model = "best.pt"

# Five Animal Classes for Detection
animal_class = [
    "cat",
    "deer",
    "dog",
    "tiger",
    "elephant",
]

# Random color generation used for boundary box plotting
colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]


@app.get("/")
# Index Page / Home Page, Renders the index.html Jinja2Template
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model": model, "animal_class": animal_class},
    )


"""
NAME: displayResuly
PARAMETERS: request, uploaded files, trained model, image size, animal class÷
PURPOSE: The function displays the result renders display_result.html detecting animals on the uploaded image
PRECONDITION: The files are uploaded, and model is trained
POSTCONDITION: It renders the Jinja2Template display_result.html
"""


@app.post("/")
# Displays result with parameters model = custom model, uploaded files and default image size of 640
# Returns HTML template display_result.html with boundary box data and confidence rate
def dispayResult(
    request: Request,
    file_list: List[UploadFile] = File(...),
    model: str = Form(...),
    img_size: int = Form(640),
    animal_class: List = animal_class,
):

    # Custom Model
    model_name = "best.pt"

    # Loading Model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name)

    # Getting image from uploaded File
    img_batch = [
        cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        for file in file_list
    ]

    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]
    results = model(img_batch_rgb, size=img_size)

    # Creating JSON of Class Name, Boundary box and Confidence
    json_results = returnResult(results, model)

    # Initializing the empty list variables
    img_str_list, classes_name, recommended_image_list, label_list = [], [], [], []

    # plot boundary boxes on the image
    for img, boundary_box_list in zip(img_batch, json_results):
        for boundary_box in boundary_box_list:
            if boundary_box["class_name"] in animal_class:
                label = f'{boundary_box["class_name"]} {boundary_box["confidence"]:.2f}'

                # Appending the boundary box and confidence rate value in the label_list variable
                label_list.append(boundary_box["class_name"])

                # calling plotBoundaryBox for detecting animal in the uploaded image
                plotBoundaryBox(
                    boundary_box["boundary_box"],
                    img,
                    label=label,
                    color=colors[int(boundary_box["class"])],
                    line_thickness=4,
                )

                # Getting animal class images from our custom dataset
                mypath = "./dataset/" + boundary_box["class_name"] + ".txt"
                dataset_file = open(mypath, "r")
                dataset_list = dataset_file.read()
                dataset_list = dataset_list.split("\n")
                classes_name.append(boundary_box["class_name"].capitalize())

                # Selecting 50 recommended image from dataset and appending it to the list
                random_class_images = random.sample(dataset_list, 50)
                recommended_image_list.append(random_class_images)

        # Appending base64image in the image string list
        img_str_list.append(base64Image(img))

    # Checking if the animal label list is empty in the uploaded files
    if not label_list:
        return templates.TemplateResponse(
            "display_result.html",
            {
                "request": request,
                "error_text": "No Animals Found in the given image",
            },
        )

    # Only listing the unique class of animals
    unique_class = list(set(classes_name))
    separate_unique_class = ", ".join(str(x) for x in unique_class)
    recommended_images = [x for xs in recommended_image_list for x in xs]

    # Getting 50 random images from the dataset
    recommended_images = random.sample(recommended_images, 50)

    # Render display_result.html jinja2Template
    return templates.TemplateResponse(
        "display_result.html",
        {
            "request": request,
            "boundary_box_image_data_zipped": zip(img_str_list, json_results),
            "recommend_image": recommended_images,
            "class_name": separate_unique_class,
            "label_list": label_list,
        },
    )


"""
NAME: returnResult
PARAMETERS: model results and our custom model model
PURPOSE: The function creates the boundary box of the given class name with the confidence rate made by our model and return JSON
PRECONDITION: model results and custom model are send as parameters and it is called by the displayResult function
POSTCONDITION: It returns the JSON of boundary box, class name, and the confidence rate
"""


def returnResult(model_results, model):
    return [
        [
            {
                "class": int(dect[5]),
                "class_name": model.model.names[int(dect[5])],
                "boundary_box": [int(x) for x in dect[:4].tolist()],
                "confidence": float(dect[4]),
            }
            for dect in result
        ]
        for result in model_results.xyxy
    ]


"""
NAME: plotBoundaryBox
PARAMETERS: x, image, color, label, line_thickness
PURPOSE: The function plots the boundary box in the detected animal class
PRECONDITION: Boundary box, image, and animal class label is needed to plot the boundary box
POSTCONDITION: It returns the  boundary box detecting the animal in the given image
"""


def plotBoundaryBox(x, image, color=(128, 128, 128), label=None, line_thickness=4):
    thickness_of_line = (
        line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    )
    color_1, color_2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(
        image,
        color_1,
        color_2,
        color,
        thickness=thickness_of_line,
        lineType=cv2.LINE_AA,
    )
    if label:
        font_thickness = max(thickness_of_line - 1, 1)
        thickness_size = cv2.getTextSize(
            label, 0, fontScale=thickness_of_line / 3, thickness=font_thickness
        )[0]
        color_2 = color_1[0] + thickness_size[0], color_1[1] - thickness_size[1] - 3
        cv2.rectangle(
            image, color_1, color_2, color, -1, cv2.LINE_AA
        )  # rectangle to fill
        cv2.putText(
            image,
            label,
            (color_1[0], color_1[1] - 2),
            0,
            thickness_of_line / 3,
            [225, 255, 255],
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )


"""
NAME: base64Image
PARAMETERS: x, image, color, label, line_thickness
PURPOSE: The function returns the uploaded image to base64 encoded string representation
PRECONDITION: Boundary box, image, and animal class label is needed to plot the boundary box and it is called from dispayResult function
POSTCONDITION: It returns the  boundary box detecting the animal in the given image
"""


def base64Image(img):
    _, img_to_arr = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(img_to_arr.tobytes()).decode("utf-8")

    return img_base64


# Main
if __name__ == "__main__":
    # parsing into Python objects
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000)
    parse_opt = parser.parse_args()

    main_app = "main:app"

    # Running the uvicorn server for browsing locally
    uvicorn.run(main_app, host=parse_opt.host, port=parse_opt.port, reload=True)
