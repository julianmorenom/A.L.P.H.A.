# Import dependencies for Computervision (cv2)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp

# From step 2
import os

# From step 3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline # allows you to build a ML pipeline
from sklearn.preprocessing import StandardScaler # Normalizes your data

# Diferent algorithms for diversifing your model training (?)
from sklearn.linear_model import LogisticRegression, RidgeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # accuracy metrics
import pickle

# Import OSC
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client


# Misc dependencies
import argparse
import random
import time
import keyboard

# GPT3 OPENAI
import json
import openai
from gpt import GPT
from gpt import Example

# import TensorFlow MobileNet-SDD v3 object detection
config_file ='./assets/ssd_mobilenet_v3/config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = './assets/ssd_mobilenet_v3/weights/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

# Load the model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load coco labels
classLabels = []
file_name = './assets/labels/darknet_coco_names.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    
# Set up the model
model.setInputSize(320, 320) ## size specs from the config file
model.setInputScale(1.0/127.5) ## 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5)) ## mobilenet => [-1,1]

#load an image
img = cv2.imread('./assets/images/person_with_dog.jpg')

# Using the model to detect classes from the index of our labels
ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.6)

# Gender detection setup
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['man ', 'woman ']

#load the network
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# setup OpenAI model generation
with open('GPT_SECRET_KEY_CW.json') as f:
    data = json.load(f)
openai.api_key = data["API_KEY"]
gpt = GPT(engine="davinci-instruct-beta-v3",
         temperature=0.9,
         max_tokens=500)

# calling the pose id model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# remeber to train a new one in the place
# open the working ID model
with open('body_language.pkl', 'rb') as f:
    poseIDmodel = pickle.load(f)

# Variables and functions definition

# text_length = 100
camera_number = 1

happy = ["Write a happy poem that is a haiku, line endings should not rhyme. The poem should contain once the words ",
        "Write a happy poem that is between six and sixteen lines long. The poem should be written in free verse form and it's line endings should not rhyme. The poem should contain once or twice the words",
        "Write a happy poem that is a villanelle in the style of Arthur Rimbaud. The poem should contain once the words ",
        "Write a happy poem that is an ode in the style of Virgina Wolf. The poem should contain once the words ",
        "Write a happy poem that is between six and sixteen lines long in the style of Quevedo. The poem should contain once the words ",
        "Write a happy poem that is between six and sixteen lines long in the style of William Shakespear. The poem should contain once the words ",
        ]

sad = ["Write a sad poem that is a haiku, line endings should not rhyme. The poem should contain once the words ",
        "Write a sad poem that is between six and sixteen lines long. The poem should be written in free verse form and it's line endings should not rhyme. The poem should contain once or twice the words",
        "Write a sad poem that is a villanelle in the style of Arthur Rimbaud. The poem should contain once the words ",
        "Write a sad poem that is an ode in the style of Virgina Wolf. The poem should contain once the words ",
        "Write a sad poem that is between six and sixteen lines long in the style of Quevedo. The poem should contain once the words ",
        "Write a sad poem that is between six and sixteen lines long in the style of William Shakespear. The poem should contain once the words ",
        ]

angry = ["Write a angry poem that is a haiku, line endings should not rhyme. The poem should contain once the words ",
        "Write a angry poem that is between six and sixteen lines long. The poem should be written in free verse form and it's line endings should not rhyme. The poem should contain once or twice the words",
        "Write a angry poem that is a villanelle in the style of Arthur Rimbaud. The poem should contain once the words ",
        "Write a angry poem that is an ode in the style of Virgina Wolf. The poem should contain once the words ",
        "Write a angry poem that is between six and sixteen lines long in the style of Quevedo. The poem should contain once the words ",
        "Write a angry poem that is between six and sixteen lines long in the style of William Shakespear. The poem should contain once the words ",
        ]

silly = ["Write a silly poem that is a haiku, line endings should not rhyme. The poem should contain once the words ",
        "Write a silly poem that is between six and sixteen lines long. The poem should be written in free verse form and it's line endings should not rhyme. The poem should contain once or twice the words",
        "Write a silly poem that is a villanelle in the style of Arthur Rimbaud. The poem should contain once the words ",
        "Write a silly poem that is an ode in the style of Virgina Wolf. The poem should contain once the words ",
        "Write a silly poem that is between six and sixteen lines long in the style of Quevedo. The poem should contain once the words ",
        "Write a silly poem that is between six and sixteen lines long in the style of William Shakespear. The poem should contain once the words ",
        ]

# Functions
def feeling_happy(happy, gender, some_object): 
    print('Generating. Please wait...')
    prompt = random.choice(happy) + gender + some_object + '.'
    # res = generator(prompt, max_length=text_length, do_sample=True, temperature=0.9)
    res = gpt.submit_request(prompt)
    global message_to_TD
    message_to_TD = (res.choices[0].text)
    print(message_to_TD)

def feeling_sad(sad, gender, some_object):
    print('Generating. Please wait...')
    prompt = random.choice(sad) + gender + some_object + '.'
    # res = generator(prompt, max_length=text_length, do_sample=True, temperature=0.9)
    res = gpt.submit_request(prompt)
    global message_to_TD
    # message_to_TD = res[0]['generated_text']
    message_to_TD = (res.choices[0].text)
    print(message_to_TD)
    
def feeling_silly(silly, gender, some_object):
    print('Generating. Please wait...')
    prompt = random.choice(silly) + gender + some_object + '.'
    # res = generator(prompt, max_length=text_length, do_sample=True, temperature=0.9)
    res = gpt.submit_request(prompt)
    global message_to_TD
    # message_to_TD = (res[0]['generated_text'])  
    message_to_TD = (res.choices[0].text)

def feeling_angry(angry, gender, some_object):
    print('Generating. Please wait...')
    prompt = random.choice(angry) + gender + some_object + '.'
    res = gpt.submit_request(prompt)
    #res = generator(prompt, max_length=text_length, do_sample=True, temperature=0.9)
    global message_to_TD
    # message_to_TD = (res[0]['generated_text'])
    message_to_TD = (res.choices[0].text)
    
def getFaceBox(net, frame,conf_threshold = 0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300),[104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]* frameWidth)
            y1 = int(detections[0,0,i,4]* frameHeight)
            x2 = int(detections[0,0,i,5]* frameWidth)
            y2 = int(detections[0,0,i,6]* frameHeight)
            bboxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)

    return frameOpencvDnn , bboxes
    
# Define average find function in a list
def most_freq(List):
    counter = 0
    list_obj = List[0]
    
    for i in List:
        curr_freq = List.count(i)
        if(curr_freq<counter):
            counter = curr_freq
            list_obj = i 
        
        return list_obj
    
# Define the finding person function   
def analyze_person():
    # Calling the model for person ID
    cap = cv2.VideoCapture(camera_number)
    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    state_completion = 0
    results = []
    
    # Run the model 
    while True:
        ret, frame = cap.read()

        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.7)
                                                    
        ##print(ClassIndex)
        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(
                ClassIndex.flatten(), confidence.flatten(), bbox
            ):
                if ClassInd <= 91:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)

                    #label, con = (classLabels[ClassInd - 1], str(confidence * 100)) 

                    cv2.putText(
                        frame,
                        #label + ": " + con,
                        classLabels[ClassInd - 1],
                        (boxes[0] + 10, boxes[1] + 40),
                        font,
                        fontScale=font_scale,
                        color=(0, 255, 0),
                        thickness=3,
                    )
                    if classLabels[ClassInd - 1] == 'person':
                        # Send OSC message to TD
                        state_completion = 1
                        
            cv2.imshow("Video", frame)
            if state_completion == 1:
                break
            if cv2.waitKey(2) & 0xFF == ord("q"):
                break
                            
    cap.release()
    cv2.destroyAllWindows()
    return state_completion
                        
# Computer vision analysis function
def computer_vision_capture():
    cap = cv2.VideoCapture(camera_number)
    emotions = ['Happy','Sad','Silly', 'Angry']
    objects = ['door']
    seq_forward_time = 8
    init_time = time.time()

    #Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor back
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80,256,121), thickness=1,circle_radius=1)
                                     )
            # Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80,256,121), thickness=2,circle_radius=2)
                                     )
            # Left hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80,256,121), thickness=2,circle_radius=2)
                                     )
            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245,117,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245,256,121), thickness=2,circle_radius=2)
                                     )
            # Export coordinates
            try:
                # Extract pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract face landmarks 
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate rows
                row = pose_row + face_row

    #             # Append class name
    #             row.insert(0, class_name)

                # Make detections
                x = pd.DataFrame([row])
                body_language_class = poseIDmodel.predict(x)[0]
                body_language_prob = poseIDmodel.predict_proba(x)[0]
                if body_language_class != 'Winning':
                    emotions.append(body_language_class)
                # print(body_language_class, body_language_prob)

                # Grab ear coordinates
                display_coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                         , [640, 480]).astype(int))

                cv2.rectangle(image,
                             (display_coords[0], display_coords[1]+5),
                             (display_coords[0]+ len(body_language_class)*20, display_coords[1]-30),
                             (245, 117, 16), -1)
                cv2.putText(image, body_language_class, display_coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            except:
                pass

            cv2.imshow('Raw webcam feed', image)

            # close the video when 30 sec have elapsed
            current_time = time.time()
            if cv2.waitKey(2) & 0xFF == ord("q"):
                break

            break_time = current_time - init_time
            print(break_time)
            if break_time > seq_forward_time:          
                break

    cap.release()
    cv2.destroyAllWindows()
    print(emotions)
    print(most_freq(emotions))
    
    ### INSERT GENDER ID 
    cap = cv2.VideoCapture(camera_number)
    padding = 20
    init_time = time.time()
    all_genders = []
    
    while cv2.waitKey(1) < 0:
        #read frame
        t = time.time()
        hasFrame , frame = cap.read()

        if not hasFrame:
            cv2.waitKey()
            break
            
        #creating a smaller frame for better optimization
        small_frame = cv2.resize(frame,(0,0),fx = 0.5,fy = 0.5)

        frameFace ,bboxes = getFaceBox(faceNet,small_frame)
        if not bboxes:
            #print("No face Detected, Checking next frame")
            continue
        for bbox in bboxes:
            face = small_frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                    max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
            
            all_genders.append(gender)
            

            label = "{}".format(gender)
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gender Detection", frameFace)     
                
            break_time = t-init_time
               
            if cv2.waitKey(2) & 0xFF == ord("q"):
                break
                
        if break_time > seq_forward_time:     
            break
       
    cap.release()
    cv2.destroyAllWindows()
    

    # Load the video
    cap = cv2.VideoCapture(camera_number)
    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    init_time = time.time()

    while True:
        ret, frame = cap.read()


        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)

        ##print(ClassIndex)
        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(
                ClassIndex.flatten(), confidence.flatten(), bbox
            ):
                if ClassInd <= 91:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)

                    #label, con = (classLabels[ClassInd - 1], str(confidence * 100)) 

                    cv2.putText(
                        frame,
                        #label + ": " + con,
                        classLabels[ClassInd - 1],
                        (boxes[0] + 10, boxes[1] + 40),
                        font,
                        fontScale=font_scale,
                        color=(0, 255, 0),
                        thickness=3,
                    )
                    if classLabels[ClassInd - 1] != 'person' and classLabels[ClassInd - 1] != 'remote':
                        if classLabels[ClassInd - 1] not in objects :
                            objects.append(classLabels[ClassInd - 1])


            cv2.imshow("Video", frame)

            # close the video when 30 sec have elapsed
            current_time = time.time()
            if cv2.waitKey(2) & 0xFF == ord("q"):
                break

            break_time = current_time-init_time
            if break_time > seq_forward_time:          
                break

    cap.release()
    cv2.destroyAllWindows()
    print(objects)
    results = [random.choice(emotions), most_freq(all_genders), random.choice(objects)]    
    return results

# function that gets the mood and calls the correct generation function
def mood_selector(results):
    print(results[0])
    if results[0] == 'Happy':
        print('we are on the happy state')
        message_to_send = feeling_happy(happy, results[1], results[2])
      
    
    if results[0] == 'Sad':
        print('we are on the sad state')
        message_to_send = feeling_sad(sad, results[1], results[2])
      
        
    if results[0] == 'Silly':
        print('we are on the silly state')
        message_to_send = feeling_silly(silly, results[1], results[2])
        
    if results[0] == 'Angry':
        print('we are on the Angry state')
        message_to_send = feeling_angry(angry, results[1], results[2])
        
        
# Server osc setup
if __name__ == "__main__":
    ip = "192.168.178.53"
    sendPort = 12000

    # Sending OSC to Processing
    client = udp_client.SimpleUDPClient(ip, sendPort)
    # Sendin printer OSC RaspPis
    client_print = udp_client.SimpleUDPClient("192.168.178.149", 5000)

state = 1
id_person = 0
capture_results = []
# send the initial state message
client.send_message("/state", state)
# main operating function
while True:
    client.send_message("/st", state)
    # call the analyse person
    if state == 1:
        print("Sate 0: find a person in image")
        id_person = analyze_person()
        if id_person:
            state = 2
            client.send_message("/st", state)
            id_person = "0"
        print(state)
    elif state == 2:
        print("State 1: Analyze persons mood, genre and object")
        capture_results = computer_vision_capture()
        print('this are the results')
        print(capture_results)
        state = 3
    elif state == 3:
        print("State 2: calling the generation functions")
        client.send_message("/st", state)
        mood_selector(capture_results)
        # This send message was neccesary as TD was not reciving only once
        # client.send_message("/messageIn", message_to_TD)
        print(message_to_TD)
        state = 4
        client.send_message("/st", state)
        # client.send_message("/st", state)
    elif state == 4:
        print("State 3: send message to screen")
        client.send_message("/st", state)
        screen_message = message_to_TD.replace('\n',' ')
        screen_message = screen_message.replace('-','')
        client.send_message("/messageIn", screen_message)
        client.send_message("/st", state)
        time.sleep(30)
        state = 5
    elif state == 5:
        print("State 4: Sending message to printer")
        client.send_message("/state", state)
        message_to_print = message_to_TD.replace('"', ' ')
        message_to_print = message_to_TD.replace('\n', '')
        client_print.send_message("/message_to_raspPi", message_to_print)
        time.sleep(45)
        state = 1
        client.send_message("/state", state)
        
    if keyboard.is_pressed('q'):
        print('You pressed quit key!')
        break  # finishing the loop