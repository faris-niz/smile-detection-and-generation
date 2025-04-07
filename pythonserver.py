# importing all the functions 
# from http.server module 
from http.server import *
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import numpy as np
import json
import cv2
import os
from PIL import Image


# creating a class for handling 
# basic Get and Post Requests 
def load_image(filename, size=(256,256)):
# load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels

def crop_face_with_cv2(image_path):
    face_cascade_path = "haarcascade_frontalface_alt2.xml"
    # Load the image
    image = cv2.imread(image_path)

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_roi = None	

    for (x, y, w, h) in faces:
        # Crop the face
        face_roi = image[y:y+h, x:x+w]
        cv2.imwrite('cropped_image.jpg', face_roi) 

        # Display the cropped face
        # cv2.imshow("Cropped Face", face_roi)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return faces[0]


def crop_image(image, left_percent, top_percent, right_percent, bottom_percent):
    """Crops an image based on the given percentages."""

    width, height = image.size
    left = int(width * left_percent / 100)
    top = int(height * top_percent / 100)
    right = int(width * (100 - right_percent) / 100)
    bottom = int(height * (100 - bottom_percent) / 100)

    return image.crop((left, top, right, bottom))


class GFG(BaseHTTPRequestHandler): 
	# creating a function for Get Request 

	def do_POST(self):

		if self.path == "/detect":
			contentlen = int(self.headers.get('Content-Length'))
			post_body = self.rfile.read(contentlen)
			jsontst = json.loads(post_body)
			src_image = load_image('face-recognition-brain-api/'+jsontst['path'], (120, 120))
			model = load_model('facetracker2.h5')
			yhat = model.predict(src_image)
			response = json.dumps(yhat[0][0].tolist()+yhat[1][0].tolist())
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(response.encode())
		else:		
			contentlen = int(self.headers.get('Content-Length'))
			post_body = self.rfile.read(contentlen)
			jsontst = json.loads(post_body)

			#crop the face from the image
			face = crop_face_with_cv2('face-recognition-brain-api/'+jsontst['path'])

			# Save the cropped image
			src_image = load_image('cropped_image.jpg')
			model = load_model('epoch15modelGeneral.h5')
			gen_image = model.predict(src_image)
			gen_image = (gen_image + 1) / 2.0
			pyplot.imsave("model_output.jpg", gen_image[0])
			bg_image = Image.open('face-recognition-brain-api/'+jsontst['path'])
			overlay_image = Image.open('model_output.jpg')
			overlay_image=overlay_image.resize((face[2], face[3]))
			bg_image.paste(overlay_image, (face[0], face[1]))
			bg_image.save('face-recognition-brain/public/smileOutput/'+jsontst['name'])
			self.send_response(200) 
			# Type of file that we are using for creating our 
			# web server. 
			self.send_header('content-type', 'text/html') 
			self.end_headers() 
			
			# what we write in this function it gets visible on our 
			# web-server 
			self.wfile.write('<h1>GFG - (GeeksForGeeks)</h1>'.encode()) 


# this is the object which take port 
# number and the server-name 
# for running the server 
portNumber = 5555
port = HTTPServer(('', portNumber), GFG) 


# this is used for running our 
# server as long as we wish 
# i.e. forever 
port.serve_forever() 
