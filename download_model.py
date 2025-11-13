import gdown

url = "https://drive.google.com/file/d/1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf/viewc"
output = "plant_disease_prediction_model.h5"
gdown.download(url, output, quiet=False)
