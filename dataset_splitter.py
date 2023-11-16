import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_datasets(input_folder1, input_folder2, output_folder, modelid,
                   test_size=0.2, validation_size=0.2, random_seed=42):
    # Crea le cartelle di output
    output_folder1 = os.path.join(output_folder, input_folder1)
    output_folder2 = os.path.join(output_folder, input_folder2)

    for folder in [output_folder1, output_folder2]:
        os.makedirs(folder, exist_ok=True)
        for subset_folder in ["Train", "Test", "Valid"]:
            os.makedirs(os.path.join(folder, subset_folder), exist_ok=True)
            for label in ["Fake", "Real"]:
                os.makedirs(os.path.join(folder, subset_folder, label), exist_ok=True)

    # Estrai la lista di file comuni dalle cartelle input_folder1 e input_folder2
    files_folder1_fake = os.listdir(os.path.join(input_folder1, "Fake"))
    files_folder1_real = os.listdir(os.path.join(input_folder1, "Real"))
    files_folder2_fake = os.listdir(os.path.join(input_folder2, "Fake"))
    files_folder2_real = os.listdir(os.path.join(input_folder2, "Real"))

    files_folder1 = set(files_folder1_fake +
                        files_folder1_real)
    files_folder2 = set(files_folder2_fake +
                        files_folder2_real)
    files_folder2 = set([file.replace(modelid, "")
                        for file in files_folder2])

    # Trova l'intersezione tra i due insiemi di file
    common_files = files_folder2.intersection(files_folder1)

    # Converte l'insieme in una lista
    common_filelist = list(common_files)

    # Suddivisione in Train, Test e Validation
    train_images, test_valid_images = train_test_split(common_filelist, test_size=(test_size + validation_size),
                                                       random_state=random_seed)
    test_images, valid_images = train_test_split(test_valid_images,
                                                 test_size=(validation_size / (test_size + validation_size)),
                                                 random_state=random_seed)

    # Sposta le immagini nelle rispettive cartelle nei due dataset
    for i in tqdm(range(len(train_images)),desc="Moving train images..."):
        image_folder1 = train_images[i]
        image_folder2 = modelid + image_folder1
        if image_folder1 in files_folder1_real:
            shutil.move(os.path.join(input_folder1, "Real", image_folder1),
                        os.path.join(output_folder1, "Train", "Real", image_folder1))
            shutil.move(os.path.join(input_folder2, "Real", image_folder2),
                        os.path.join(output_folder2, "Train", "Real", image_folder2))
        else:
            shutil.move(os.path.join(input_folder1, "Fake", image_folder1),
                        os.path.join(output_folder1, "Train", "Fake", image_folder1))
            shutil.move(os.path.join(input_folder2, "Fake", image_folder2),
                        os.path.join(output_folder2, "Train", "Fake", image_folder2))


    for i in tqdm(range(len(test_images)), desc="Moving test images..."):
        image_folder1 = test_images[i]
        image_folder2 = modelid + image_folder1
        if image_folder1 in files_folder1_real:
            shutil.move(os.path.join(input_folder1, "Real", image_folder1),
                        os.path.join(output_folder1, "Test", "Real", image_folder1))
            shutil.move(os.path.join(input_folder2, "Real", image_folder2),
                        os.path.join(output_folder2, "Test", "Real", image_folder2))
        else:
            shutil.move(os.path.join(input_folder1, "Fake", image_folder1),
                        os.path.join(output_folder1, "Test", "Fake", image_folder1))
            shutil.move(os.path.join(input_folder2, "Fake", image_folder2),
                        os.path.join(output_folder2, "Test", "Fake", image_folder2))

    for i in tqdm(range(len(valid_images)), desc="Moving validation images..."):
        image_folder1 = valid_images[i]
        image_folder2 = modelid + image_folder1
        if image_folder1 in files_folder1_real:
            shutil.move(os.path.join(input_folder1, "Real", image_folder1),
                        os.path.join(output_folder1, "Valid", "Real", image_folder1))
            shutil.move(os.path.join(input_folder2, "Real", image_folder2),
                        os.path.join(output_folder2, "Valid", "Real", image_folder2))
        else:
            shutil.move(os.path.join(input_folder1, "Fake", image_folder1),
                        os.path.join(output_folder1, "Valid", "Fake", image_folder1))
            shutil.move(os.path.join(input_folder2, "Fake", image_folder2),
                        os.path.join(output_folder2, "Valid", "Fake", image_folder2))


# Imposta i percorsi delle cartelle
input_folder1 = "Dataset_seg"
input_folder2 = "Dataset_mtcnn"
output_folder = "Dataset_faces_only_ready_to_train"
modelid = "mtcnn_"

# Suddividi e sposta i file nei due dataset
split_datasets(input_folder1, input_folder2, output_folder, modelid)
