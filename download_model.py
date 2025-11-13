import gdown

url = "https://drive.google.com/uc?id=1LUHc8cwqjYfZMa6oVEdZBojMxq7iu6Mb"
output = "model.tflite" 

print("Downloading model from Google Drive...")
gdown.download(url, output, quiet=False)
print("Download complete! File saved as:", output)
