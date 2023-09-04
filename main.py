import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from tensorflow.keras.models import load_model

def getPrediction(filename):
    model_path = 'model/model3.h5'
    my_model = load_model(model_path, compile=False)
    
    img_path = 'static/image/' + filename
    
    # Open the image using rasterio
    with rasterio.open(img_path) as dataset:
        # Read all bands (assuming it's a multi-band image)
        img = dataset.read()
    
    # Scale pixel values using min-max scaling
    img = (img - img.min()) / (img.max() - img.min())
    
    # Transpose the image to match the channel order expected by Keras (channels last)
    img = np.transpose(img, (1, 2, 0))
    
    img = np.expand_dims(img, axis=0)  # Get it ready as input to the network
    
    # Predict
    threshold = 0.5
    pred = my_model.predict(img)
    pred = (pred > threshold).astype(np.uint8)
    
    # Display the predicted image using matplotlib
    plt.figure(figsize=(8, 8))
    plt.title('Segmentation Result')
    plt.imshow(pred[0, :, :, 0], cmap='gray')  # Assuming it's binary segmentation
    
    plt.show()  # Display the plot
    
    return pred


