from fastapi import FastAPI, UploadFile, File, HTTPException
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from numpy import expand_dims
from matplotlib import pyplot
import numpy as np
import cv2
import os
from PIL import Image
import json

app = FastAPI()


def load_image(filename, size=(256, 256)):
    pixels = load_img(filename, target_size=size)
    pixels = img_to_array(pixels)
    pixels = (pixels - 127.5) / 127.5
    pixels = expand_dims(pixels, 0)
    return pixels


def crop_face_with_cv2(image_path):
    face_cascade_path = "haarcascade_frontalface_alt2.xml"
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise HTTPException(status_code=404, detail="No face detected")
    face_roi = None
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        cv2.imwrite('cropped_image.jpg', face_roi)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return faces[0]


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    src_image = load_image(file.filename, (120, 120))
    model = load_model('facetracker2.h5')
    yhat = model.predict(src_image)
    response = yhat[0][0].tolist() + yhat[1][0].tolist()
    return json.dumps(response)


@app.post("/")
async def generate(file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    face = crop_face_with_cv2(file.filename)
    src_image = load_image('cropped_image.jpg')
    model = load_model('epoch15modelGeneral.h5')
    gen_image = model.predict(src_image)
    gen_image = (gen_image + 1) / 2.0
    pyplot.imsave("model_output.jpg", gen_image[0])
    bg_image = Image.open(file.filename)
    overlay_image = Image.open('model_output.jpg')
    overlay_image = overlay_image.resize((face[2], face[3]))
    bg_image.paste(overlay_image, (face[0], face[1]))
    output_path = f'face-recognition-brain/public/smileOutput/{file.filename}'
    bg_image.save(output_path)
    return {"output_path": output_path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5555)
