import os
import cv2
import numpy as np
from tqdm import tqdm
from retinaface import RetinaFace
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import gc

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def detect_faces(image_path, detector):
    faces = detector.detect_faces(img_path=image_path)

    if isinstance(faces, tuple) and len(faces) == 2:
        # Tuple con due array vuoti, non ci sono volti rilevati
        print(f"No faces detected in {image_path}")
        return {}

    print(f"Original faces: {faces}")

    # Converti le coordinate facciali in interi
    for face_key, face_info in faces.items():
        face_info['facial_area'] = list(map(int, face_info['facial_area']))

    return faces


def align_faces(image, faces, segmentation_model):
    aligned_faces = []
    for face_key, face_info in faces.items():
        x, y, w, h = face_info['facial_area']
        face_image = image[y:y + h, x:x + w]

        # Applica la segmentazione umana
        segmented_image = segment_human(face_image, segmentation_model)

        aligned_faces.append(segmented_image)
    return aligned_faces


def segment_human(input_img, segmentation_model):
    image = Image.fromarray(input_img)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    input_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = segmentation_model(input_image)['out'][0]
        output_predictions = output.argmax(0)

    person_mask = (output_predictions == 15).numpy().astype(np.uint8)
    person_mask = cv2.resize(person_mask, (image.width, image.height))
    segmented_image = np.array(image) * np.expand_dims(person_mask, axis=-1)

    return segmented_image

def main():
    segmentation_model = deeplabv3_resnet101(pretrained=True)
    segmentation_model = segmentation_model.eval()

    face_detector = RetinaFace

    dataset_path = "Dataset_merged"
    output_path = "Dataset_retinaface"

    for label in ["Real", "Fake"]:
        label_path = os.path.join(dataset_path, label)
        output_label_path = os.path.join(output_path, label)

        os.makedirs(output_label_path, exist_ok=True)

        for filename in tqdm(os.listdir(label_path), desc=f"Processing {label} images"):
            image_path = os.path.join(label_path, filename)

            output_filename = f"{filename.split('.')[0]}_face0.png"
            output_face_path = os.path.join(output_label_path, output_filename)

            if os.path.exists(output_face_path):
                continue

            img = load_and_preprocess_image(image_path)
            faces = detect_faces(image_path, face_detector)
            aligned_faces = align_faces(img, faces, segmentation_model)

            i = 0
            for face_key, face_info in faces.items():
                face_image = aligned_faces[i]
                output_filename = f"{filename.split('.')[0]}_face{i}.png"
                output_face_path = os.path.join(output_label_path, output_filename)
                cv2.imwrite(output_face_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                i += 1

            del faces, aligned_faces
            gc.collect()

if __name__ == "__main__":
    main()

