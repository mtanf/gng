import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def detect_faces(image_path):
    image = load_and_preprocess_image(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return faces

def segment_human(image_path, segmentation_model):
    image = Image.fromarray(load_and_preprocess_image(image_path))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    input_image = transform(image).unsqueeze(0)

    # Perform segmentation using DeepLabV3
    with torch.no_grad():
        output = segmentation_model(input_image)['out'][0]
        output_predictions = output.argmax(0)

    # Create a binary mask indicating where people are (assuming class 15 corresponds to person)
    person_mask = (output_predictions == 15).numpy().astype(np.uint8)

    # Use the binary mask to segment the person from the original image
    segmented_image = np.array(image) * np.expand_dims(person_mask, axis=-1)

    return segmented_image

def main():
    # Load pre-trained segmentation model (e.g., DeepLabV3)
    segmentation_model = deeplabv3_resnet101(pretrained=True)
    segmentation_model = segmentation_model.eval()

    dataset_path = "Dataset_merged"
    output_path = "Dataset_seg"

    for label in ["Real", "Fake"]:
        label_path = os.path.join(dataset_path, label)
        output_label_path = os.path.join(output_path, label)

        os.makedirs(output_label_path, exist_ok=True)

        for filename in os.listdir(label_path):
            image_path = os.path.join(label_path, filename)

            # Detect faces using MTCNN
            faces = detect_faces(image_path)

            # Perform human segmentation
            segmented_image = segment_human(image_path, segmentation_model)

            # Combine face detection and human segmentation
            for face in faces:
                x, y, w, h = face['box']
                face_image = segmented_image[y:y+h, x:x+w]
                output_filename = f"{filename.split('.')[0]}_face{len(os.listdir(output_label_path))}.png"
                output_face_path = os.path.join(output_label_path, output_filename)
                cv2.imwrite(output_face_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
