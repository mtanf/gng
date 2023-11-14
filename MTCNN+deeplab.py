import os
import cv2
import numpy as np
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import gc
from memory_profiler import profile


def load_and_preprocess_image(image_path):
    image = None
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def detect_faces(image_path, detector, enlarge_percentage=30):
    image = load_and_preprocess_image(image_path)

    faces = detector.detect_faces(image)

    # Enlarge bounding boxes by a certain percentage
    for face in faces:
        x, y, w, h = face['box']
        width_increase = int(w * (enlarge_percentage / 100.0))
        height_increase = int(h * (enlarge_percentage / 100.0))

        # Limitando le coordinate alle dimensioni dell'immagine
        x = max(0, x - width_increase // 2)
        y = max(0, y - height_increase // 2)
        w = min(image.shape[1] - x, w + width_increase)
        h = min(image.shape[0] - y, h + height_increase)

        face['box'] = [x, y, w, h]
    del image
    gc.collect()
    return faces

def segment_human(image_path, segmentation_model):
    image = None
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

    # Resize the binary mask to the size of the original image
    person_mask = cv2.resize(person_mask, (image.width, image.height))

    # Use the binary mask to segment the person from the original image
    segmented_image = np.array(image) * np.expand_dims(person_mask, axis=-1)

    del image, input_image, output, output_predictions, person_mask
    gc.collect()
    torch.cuda.empty_cache()
    return segmented_image


@profile
def main():
    # Load pre-trained segmentation model (e.g., DeepLabV3)
    segmentation_model = deeplabv3_resnet101(pretrained=True)
    segmentation_model = segmentation_model.eval()
    face_detector = MTCNN(min_face_size=80)

    dataset_path = "Dataset_merged"
    output_path = "Dataset_seg"
    output_path_mtcnn = "Dataset_mtcnn"

    for label in ["Real", "Fake"]:
        label_path = os.path.join(dataset_path, label)
        output_label_path = os.path.join(output_path, label)
        output_label_path_mtcnn = os.path.join(output_path_mtcnn, label)


        os.makedirs(output_label_path, exist_ok=True)
        os.makedirs(output_label_path_mtcnn, exist_ok=True)

        for filename in tqdm(os.listdir(label_path), desc=f"Processing {label} images"):
            image_path = os.path.join(label_path, filename)

            # Verify if the image has already been processed
            output_filename = f"{filename.split('.')[0]}_face0.png"
            output_face_path = os.path.join(output_label_path, output_filename)

            if os.path.exists(output_face_path):
                # print(f"Image {filename} already processed. Skipping.")
                continue

            # Detect faces using MTCNN
            faces = detect_faces(image_path, face_detector)

            # Perform human segmentation
            segmented_image = segment_human(image_path, segmentation_model)


            # Combine face detection and human segmentation
            i=0
            # cv2.imshow("segmented image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
            for face in faces:
                x, y, w, h = face['box']
                face_image = segmented_image[y:y+h, x:x+w]
                face_image_mtcnn = load_and_preprocess_image(image_path)[y:y+h, x:x+w]
                output_filename = f"{filename.split('.')[0]}_face{i}.png"
                output_face_path = os.path.join(output_label_path, output_filename)
                output_face_path_mtcnn = os.path.join(output_label_path_mtcnn, "mtcnn_"+output_filename)
                cv2.imwrite(output_face_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(output_face_path_mtcnn, cv2.cvtColor(face_image_mtcnn, cv2.COLOR_RGB2BGR))
                # cv2.imshow("final_face_mtcnn{}".format(i), cv2.cvtColor(face_image_mtcnn, cv2.COLOR_RGB2BGR))
                # cv2.imshow("final_face{}".format(i), cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                i+=1
            del faces, segmented_image
            gc.collect()

            # cv2.waitKey(0)
    del segmentation_model, face_detector
    gc.collect()

if __name__ == "__main__":
    main()
