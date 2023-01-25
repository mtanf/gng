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

arguments = parser.parse_args()

original_folder = arguments.original_folder
destination_folder = arguments.destination_folder
offset = int(arguments.bounding_box_offset)
num_chunks = int(arguments.num_chunks)


if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

f = []
for (dirpath, dirnames, filenames) in os.walk(original_folder):
    f.extend(filenames)
    break

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
        result = detector.detect_faces(frame)
        # cv2.putText(frame, "Faces found: {}".format(len(result)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (0, 255, 0), 1, cv2.LINE_AA)
        for i in range(len(result)):
            face = result[i]
            bounding_box = face['box']
            x1 = bounding_box[0] - offset
            y1 = bounding_box[1] - offset
            
            x2 =  bounding_box[0]
            y2 =  bounding_box[1]
            w = bounding_box[2] + offset
            h = bounding_box[3] + offset
    
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            try:
                roi_color = frame[y1:y2 + h, x1:x2 + w]
                cv2.imwrite(os.path.join(destination_folder, img_name+"_OF_{}_".format(offset) +str(i)+".jpg"),roi_color)
            except:
                print("Something went terribly wrong with this face, skipping")
        # cv2.imshow("Captured frame", frame)
        # try:
        #     cv2.imshow("Roi", roi_color)
        # except:
        #     continue
