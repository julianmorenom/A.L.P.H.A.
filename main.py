########## A.L.P.H.A. ########
#
# Author: Julián Moreno
# Date: 05/06/22
# Description: 
# This programm is an installation usign gpt-3 and tensorflow lite, creates poetry based 
# on what it sees. 
# 
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
import keyboard
import importlib.util
import random
from threading import Thread
from pythonosc import udp_client

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(320,240),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]

    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])

            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)

    return frameOpencvDnn,faceBoxes

# Define list of promts
def define_promt_list():
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
    return happy, sad, angry, silly

def most_freq(List):
    counter = 0
    list_obj = List[0]
    
    for i in List:
        curr_freq = List.count(i)
        if(curr_freq<counter):
            counter = curr_freq
            list_obj = i 
        
        return list_obj

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

def analyze_person():
    state_completion = 0

    # Initialize frame rate calculation and video stream
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    while state_completion == 0:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                if(object_name == 'person'):
                    state_completion = 1

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # todo: check if we can get rid of this 
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()


def computer_vision_capture():
    # Initialize frame rate calculation and video stream
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)
    # variables 
    init_time = time.time()
    seq_forward_time = 8
    break_time = 0
    emotions = ['Happy', 'Sad', 'Silly', 'Angry']
    all_genders = []
    objects = []

    while break_time < seq_forward_time:
        hasFrame, frame = videostream.read()
        resultImg,faceBoxes=highlightFace(faceNet,frame)

        # time management
        current_time = time.time()
        break_time = current_time - init_time

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            all_genders.append(gender)

            text = "{}".format(gender)
            cv2.putText(resultImg, text,(faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break 
    cv2.destroyAllWindows()
    videostream.stop()
    
    # object ID part
    second_init_time = time.time()
    second_break_time = 0
    freq = cv2.getTickFrequency()
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    while second_break_time < seq_forward_time:
        # time management
        current_time = time.time()
        second_break_time = current_time - second_break_time

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                if(object_name != 'person'):
                    if object_name not in objects:
                            objects.append(object_name)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # todo: check if we can get rid of this 
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()

    # choose results and return
    print(objects)    
    results = [random.choice(emotions), most_freq(all_genders), random.choice(objects)]
    return results

# setup the gender variables
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt" 
genderModel = "gender_net.caffemodel" 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male','Female']
faceNet = cv2.dnn.readNet(faceModel,faceProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)
# video=cv2.VideoCapture(0)
padding=20

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# define globals
happy, sad, angry, silly = define_promt_list()
state = 1
id_person = 0
capture_results = []

#check the ip ad before running
client_processing = udp_client.SimpleUDPClient("192.168.178.53", 12000)
client_print = udp_client.SimpleUDPClient("192.168.178.149", 5000)


# send the initial state message
client_processing.send_message("/state", state)

while True:
    client_processing.send_message("/st", state)
    # call the analyse person
    if state == 1:
        print("State 0: find a person in image")
        id_person = analyze_person()
        if id_person:
            state = 2
            client_processing.send_message("/st", state)
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
        client_processing.send_message("/st", state)
        mood_selector(capture_results)
        print(message_to_TD)
        state = 4
        client_processing.send_message("/st", state)
        # client.send_message("/st", state)
    elif state == 4:
        print("State 3: send message to screen")
        client_processing.send_message("/st", state)
        screen_message = message_to_TD.replace('\n',' ')
        screen_message = screen_message.replace('-','')
        client_processing.send_message("/messageIn", screen_message)
        client_processing.send_message("/st", state)
        time.sleep(30)
        state = 5
    elif state == 5:
        print("State 4: Sending message to printer")
        client_processing.send_message("/state", state)
        message_to_print = message_to_TD.replace('"', ' ')
        message_to_print = message_to_TD.replace('\n', '')
        client_print.send_message("/message_to_raspPi", message_to_print)
        time.sleep(45)
        state = 1
        client_processing.send_message("/state", state)
        
    if keyboard.is_pressed('q'):
        print('You pressed quit key!')
        break  # finishing the loop
