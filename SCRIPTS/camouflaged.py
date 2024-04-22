import cv2
import numpy as np
import csv
import os

# Define the list of object classes
classes = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load the object detection model
model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 
                                 'MobileNetSSD_deploy.caffemodel')

# List of input image filenames
image_filenames = [
    "../DATA/Test_Non_Camouflaged/non_camouflaged_bicycle.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_bird.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_bird(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_bird(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_bison.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(3).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(4).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(5).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(6).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(7).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(8).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(9).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(10).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(11).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(12).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(13).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(14).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(15).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(16).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(17).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(18).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(19).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(20).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(21).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(22).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cat(23).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cow.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cow(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cow(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_cows.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(3).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(4).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(5).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(6).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(7).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(8).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(9).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(10).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(11).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(12).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(13).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(14).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_dog(15).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant(3).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant(4).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant(5).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant(6).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephant(7).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_elephants.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(3).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(4).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(5).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(6).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(7).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(8).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_giraffe(9).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_horse.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_polar_bear.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_sheep.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_sheep(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_sheep(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_toy_animals.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra.jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(1).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(2).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(3).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(4).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(5).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(6).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(7).jpg",
"../DATA/Test_Non_Camouflaged/non_camouflaged_zebra(8).jpg",

"../DATA/Test_Camouflaged/camourflage_bat.jpg",
"../DATA/Test_Camouflaged/camourflage_bird.jpg",
"../DATA/Test_Camouflaged/camourflage_bird(1).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(2).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(3).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(4).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(5).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(6).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(7).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(8).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(9).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(10).jpg",
"../DATA/Test_Camouflaged/camourflage_bird(11).jpg",
"../DATA/Test_Camouflaged/camourflage_butterfly.jpg",
"../DATA/Test_Camouflaged/camourflage_cat.jpg",
"../DATA/Test_Camouflaged/camourflage_caterpillar.jpg",
"../DATA/Test_Camouflaged/camourflage_chameleon.jpg",
"../DATA/Test_Camouflaged/camourflage_chameleon(1).jpg",
"../DATA/Test_Camouflaged/camourflage_cheetah.jpg",
"../DATA/Test_Camouflaged/camourflage_crab.jpg",
"../DATA/Test_Camouflaged/camourflage_crocodile.jpg",
"../DATA/Test_Camouflaged/camourflage_cuttlefish.jpg",
"../DATA/Test_Camouflaged/camourflage_dog.jpg",
"../DATA/Test_Camouflaged/camourflage_dog(1).jpg",
"../DATA/Test_Camouflaged/camourflage_dog(2).jpg",
"../DATA/Test_Camouflaged/camourflage_dog(3).jpg",
"../DATA/Test_Camouflaged/camourflage_dog(4).jpg",
"../DATA/Test_Camouflaged/camourflage_duck.jpg",
"../DATA/Test_Camouflaged/camourflage_fern.jpg",
"../DATA/Test_Camouflaged/camourflage_fish.jpg",
"../DATA/Test_Camouflaged/camourflage_fish(1).jpg",
"../DATA/Test_Camouflaged/camourflage_fish(2).jpg",
"../DATA/Test_Camouflaged/camourflage_fish(3).jpg",
"../DATA/Test_Camouflaged/camourflage_fish(4).jpg",
"../DATA/Test_Camouflaged/camourflage_flounder.jpg",
"../DATA/Test_Camouflaged/camourflage_flounder(1).jpg",
"../DATA/Test_Camouflaged/camourflage_flounder(2).jpg",
"../DATA/Test_Camouflaged/camourflage_fox.jpg",
"../DATA/Test_Camouflaged/camourflage_fox(1).jpg",
"../DATA/Test_Camouflaged/camourflage_frog.jpg",
"../DATA/Test_Camouflaged/camourflage_frog(1).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(2).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(3).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(4).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(5).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(6).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(7).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(8).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(9).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(10).jpg",
"../DATA/Test_Camouflaged/camourflage_frog(11).jpg",
"../DATA/Test_Camouflaged/camourflage_giraffe.jpg",
"../DATA/Test_Camouflaged/camourflage_grasshopper.jpg",
"../DATA/Test_Camouflaged/camourflage_insect.jpg",
"../DATA/Test_Camouflaged/camourflage_insect(1).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(2).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(3).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(4).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(5).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(6).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(7).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(8).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(9).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(10).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(11).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(12).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(13).jpg",
"../DATA/Test_Camouflaged/camourflage_insect(14).jpg",
"../DATA/Test_Camouflaged/camourflage_lion.jpg",
"../DATA/Test_Camouflaged/camourflage_lizard.jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(1).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(2).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(3).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(4).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(5).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(6).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(7).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(8).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(9).jpg",
"../DATA/Test_Camouflaged/camourflage_lizard(10).jpg",
"../DATA/Test_Camouflaged/camourflage_moth.jpg",
"../DATA/Test_Camouflaged/camourflage_moth(1).jpg",
"../DATA/Test_Camouflaged/camourflage_moth(2).jpg",
]  

# Write the results to a CSV file
with open('../DATA/results.csv', 'w', newline='', encoding="utf8") as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["filename","camouflaged","object_detected","label", "box", "match"])
        

    # Process each input image
    for filename in image_filenames:

        # Load the input image
        image = cv2.imread(filename)
        
        # Preprocess the input image
        resized_image = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_image, 0.007843, (300, 300), 127.5)
        
        # Pass the image through the model
        model.setInput(blob)
        detections = model.forward()
        
        # Set if object is detected
        object_detected = False

        # Set if object is camouflaged
        if "non_camouflaged" in filename:
            camouflaged = False
        else:
            camouflaged = True

        # Visualize the output
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                object_detected = True
                class_id = int(detections[0, 0, i, 1])
                class_name = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype('int')
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = '{}: {:.2f}%'.format(class_name, confidence * 100)
                cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if class_name in filename:
                    match = True
                else: 
                    match = False

        # Write the results to the CSV file
        file_name = os.path.basename(filename)
        data = [file_name, str(camouflaged), str(object_detected), label, str(box), str(match)]
        writer.writerow(data)

        # Display the result
        cv2.imshow('Object Detection', image)
        cv2.waitKey(1000)  # Display each image for 0.5 seconds
        cv2.destroyAllWindows()

