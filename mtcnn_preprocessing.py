import cv2
import os
import mtcnn
import argparse as ap
import math

def split_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]
        
detector = mtcnn.MTCNN(min_face_size = 90)

parser = ap.ArgumentParser()
parser.add_argument("-orig", "--original_folder", help="Original image folder, please provide absolute paths!", required=True)
parser.add_argument("-dest", "--destination_folder", help="Destination image folder, please provide absolute paths!", required=True)
parser.add_argument("-bbo", "--bounding_box_offset", default = 20, help = "Increase/decrease MTCNN ROI boundaries", required = False)
parser.add_argument("-nc", "--num_chunks", default = 20, help = "Number of chunks for processing", required = False)
parser.add_argument("-ext", "--img_extention", default = ".jpg", help = "Extention of input files. Output files will be in .jpg. Default = '.jpg")
arguments = parser.parse_args()

original_folder = arguments.original_folder
destination_folder = arguments.destination_folder
offset = int(arguments.bounding_box_offset)
num_chunks = int(arguments.num_chunks)
ext = arguments.img_extention


with open('mtcnn_log_folder_{}.txt'.format(os.path.basename(destination_folder)), 'a') as log:

    log.write('Original files folder path: {}\n'.format(original_folder))
    log.write("Destination folder: {}\n".format(destination_folder))
    log.write("Excluded files (either MTCNN did not find anything or errors occurred):\n")
    log.close()
    
f = []
for (dirpath, dirnames, filenames) in os.walk(original_folder):
    f.extend(filenames)
    break

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
else:
    k = []
    for (dirpath, dirnames, filenames) in os.walk(destination_folder):
        k.extend(filenames)
        break
    
    for item in k:
        f.remove(item.split(".")[0]+ext)
    #f = [x for x in f if x.split(".")[0] + ".png" not in k]

imgs_list = []
for item in f:
    file_path = os.path.join(os.sep, original_folder + os.sep, item)
    imgs_list.append(file_path)


num_imgs = len(imgs_list)
chunk_size = math.floor(num_imgs/num_chunks)

for chunk in split_list(imgs_list, chunk_size):
    #load all images
    imgs = []
    img_names = []
    
    for img_path in chunk:
        try:
            tmp =cv2.imread(img_path)
            name = os.path.basename(img_path)
            img_names.append(name.split(".")[0])
            imgs.append(tmp)
        except:
            print("Could not load img from {}".format(img_path))
            continue
        
    for i in range(len(imgs)):
        frame = imgs[i]
        img_name = img_names[i]
        try:
            result = detector.detect_faces(frame)
        except:
            print("Something went terribly wrong with this face, with MTCNN, skipping")
            result = []
            
        
        if len(result)>1:
            result = [result[0]]
        elif len(result) == 0:
            with open('mtcnn_log_folder_{}.txt'.format(os.path.basename(destination_folder)), 'a') as log:
                log.write(img_name)
                log.write("\n")
                log.close()
                        
        for i in range(len(result)):
            
            face = result[i]
            bounding_box = face['box']
            confidence = face["confidence"]
            x1 = bounding_box[0] - offset
            y1 = bounding_box[1] - offset
            
            x2 =  bounding_box[0]
            y2 =  bounding_box[1]
            w = bounding_box[2] + offset
            h = bounding_box[3] + offset
            area = w*h
            
            if area > 75000:
                try:
                    roi_color = frame[y1:y2 + h, x1:x2 + w]
                    #cv2.imwrite(os.path.join(destination_folder, img_name+"_OF_{}_".format(offset) +str(i)+".jpg"),roi_color)
                    cv2.imwrite(os.path.join(destination_folder, img_name + ".jpg"),roi_color)
                except:
                    print("Something went terribly wrong with this face, skipping")
            else:
                print("Face area is too small, probable false positive, skipping")
                # roi_color = frame[y1:y2 + h, x1:x2 + w]
                # cv2.imshow("", roi_color)
                # cv2.waitKey(0)
    
    del img_names
    del imgs