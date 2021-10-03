from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from starlette.responses import RedirectResponse
import tensorflow as tf
import os

model_version = max([int(i) for i in os.listdir("server/models/") + [0]])

MODEL = tf.keras.models.load_model("server/models/"+str(model_version))
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app_desc = """<h2>Try this api by uploading potato leaf images with `/predict/`</h2>
<br>created by: Noushad Bhuiyan"""

app = FastAPI(title='Potato Disease Detector', description=app_desc)

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
