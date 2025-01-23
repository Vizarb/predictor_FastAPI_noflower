from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
MODEL_PATH = "C:/Temp_D/Progreming_Lerning/AI_ML/TensorFlowp/converted_keras/keras_model.h5"
LABELS_PATH = "C:/Temp_D/Progreming_Lerning/AI_ML/TensorFlowp/converted_keras/labels.txt"

model = load_model(MODEL_PATH, compile=False)

# Load the labels
with open(LABELS_PATH, "r") as file:
    class_names = file.readlines()


def predict_image(image_path: str):
    """
    Predicts the class of the given image using the pre-trained Keras model.
    Args:
        image_path (str): Path to the image file.
    Returns:
        (str, float): Predicted class name and confidence score.
    """
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score
