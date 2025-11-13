import gdown

# Direct link for gdown
url = "https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"

output = "plant_disease_prediction_model.h5"

print("Downloading model...")
gdown.download(url, output, quiet=False)
print("Model downloaded successfully!")
