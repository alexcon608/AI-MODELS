# import torch
# from facenet_pytorch import InceptionResnetV1
# import cv2
# import numpy as np
# import requests
# from bs4 import BeautifulSoup
# import urllib
# import os

# # Load Pretrained FaceNet model
# model = InceptionResnetV1(pretrained='vggface2').eval()

# # Function to extract face embeddings
# def extract_face_embedding(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error reading image: {image_path}")
#         return None
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_rgb = cv2.resize(img_rgb, (160, 160))
#     img_rgb = img_rgb / 255.0

#     tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float()

#     with torch.no_grad():
#         embedding = model(tensor)

#     print(f"Embedding for {image_path}: {embedding.squeeze().numpy()}")
#     return embedding.squeeze().numpy()

# # Search for images on Google
# def search_images(query, num_images=10):
#     headers = {
#         "User-Agent": "Mozilla/5.0"
#     }
#     url = f"https://www.google.com/search?q={query}&tbm=isch"
#     response = requests.get(url, headers=headers)
#     soup = BeautifulSoup(response.text, 'html.parser')

#     image_urls = []
#     for img_tag in soup.find_all('img', limit=num_images):
#         img_url = img_tag.get('src')
#         if img_url and img_url.startswith('http'):
#             image_urls.append(img_url)
    
#     print("Image URLs found: ", image_urls)
#     return image_urls

# # Helper function to download images from URL
# def download_image(url, folder='downloads'):
#     os.makedirs(folder, exist_ok=True)
#     filename = os.path.join(folder, url.split('/')[-1])
#     urllib.request.urlretrieve(url, filename)
#     return filename

# # Match uploaded image with images from the web
# def match_images(uploaded_image_path):
#     uploaded_embedding = extract_face_embedding(uploaded_image_path)
#     if uploaded_embedding is None:
#         return []  # Return empty if the image is not processed successfully
    
#     image_urls = search_images("person name", num_images=10)
    
#     matches = []
#     for url in image_urls:
#         try:
#             img_path = download_image(url)  # Download image from the URL
#             img_embedding = extract_face_embedding(img_path)
            
#             if img_embedding is None:
#                 continue
            
#             # Cosine Similarity
#             similarity = np.dot(uploaded_embedding, img_embedding) / (
#                 np.linalg.norm(uploaded_embedding) * np.linalg.norm(img_embedding)
#             )
            
#             print(f"Uploaded Embedding: {uploaded_embedding}")
#             print(f"Image Embedding: {img_embedding}")
#             print(f"Similarity: {similarity}")  # Debugging
            
#             if similarity > 0.7:  # Adjusted threshold
#                 matches.append(url)
#         except Exception as e:
#             print(f"Error processing image: {e}")
    
#     return matches

# # Main code to capture image and match
# def capture_image_from_webcam():
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         cv2.imshow("Webcam - Press 's' to capture", frame)

#         # Capture on pressing 's'
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             uploaded_image_path = 'captured_image.jpg'
#             cv2.imwrite(uploaded_image_path, frame)
#             print("Snapshot taken!")
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     # Match images with captured image
#     matches = match_images(uploaded_image_path)
#     if matches:
#         print("Matches found:", matches)
#     else:
#         print("No matches found.")

# capture_image_from_webcam()
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
import urllib
import os
from PIL import Image
from io import BytesIO
import re

# Load Pretrained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=20)  # Face detection model

# Function to extract face embeddings
def extract_face_embedding(image_path):
    img = Image.open(image_path)
    
    # Detect face and preprocess
    face = mtcnn(img)
    if face is None:
        print(f"No face detected in {image_path}")
        return None
    
    with torch.no_grad():
        embedding = model(face.unsqueeze(0))

    return embedding.squeeze().numpy()

# Search for images on Google (may need Selenium for better results)
def search_images(query, num_images=10):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    url = f"https://www.google.com/search?q={query}&tbm=isch"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    image_urls = []
    for img_tag in soup.find_all('img', limit=num_images):
        img_url = img_tag.get('src')
        if img_url and img_url.startswith("http"):
            image_urls.append(img_url)
    
    return image_urls

# Helper function to download images
def download_image(url, folder='downloads'):
    os.makedirs(folder, exist_ok=True)
    
    filename = os.path.join(folder, re.sub(r'[^\w\s-]', '', url.split('/')[-1]) + '.jpg')
    
    try:
        urllib.request.urlretrieve(url, filename)
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

# Match uploaded image with online images
def match_images(uploaded_image_path, search_term="person face"):
    uploaded_embedding = extract_face_embedding(uploaded_image_path)
    if uploaded_embedding is None:
        return []

    image_urls = search_images(search_term, num_images=10)
    matches = []

    for url in image_urls:
        img_path = download_image(url)
        if img_path is None:
            continue

        img_embedding = extract_face_embedding(img_path)
        if img_embedding is None:
            continue

        # Cosine Similarity
        similarity = np.dot(uploaded_embedding, img_embedding) / (
            np.linalg.norm(uploaded_embedding) * np.linalg.norm(img_embedding)
        )

        if similarity > 0.7:
            matches.append(url)

    return matches

# Function to handle CLI input
def main():
    print("1. Upload an image file")
    print("2. Provide an image URL")

    choice = input("Enter 1 or 2: ")

    if choice == '1':
        image_path = input("Enter the image file path: ")
        if os.path.exists(image_path):
            matches = match_images(image_path)
            print("Matches found:", matches if matches else "No matches found.")
        else:
            print("Invalid file path.")

    elif choice == '2':
        image_url = input("Enter the image URL: ")
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img_path = 'downloaded_image.jpg'
            img.save(img_path)
            matches = match_images(img_path)
            print("Matches found:", matches if matches else "No matches found.")
        except Exception as e:
            print(f"Error downloading image: {e}")

    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == '__main__':
    main()
