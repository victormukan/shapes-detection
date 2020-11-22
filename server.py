import cv2
from flask import Flask, request
import os
import src.recognition as rec
import src.s3 as s3
from uuid import uuid4

BUCKET = os.environ.get("AWS_BUCKET")


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
  if 'file' not in request.files:
    return {
      "body": "error"
    }
  
  file = request.files['file']
  file.save(os.path.join('./tmp/', "image.jpg"))

  img = cv2.imread("tmp/image.jpg", cv2.IMREAD_COLOR)

  id = str(uuid4())

  circles = rec.detect_circles(img)
  cv2.imwrite("tmp/circles.jpg", circles)

  lines = rec.detect_lines_and_intersections(img)
  cv2.imwrite("tmp/lines.jpg", lines)
  
  s3.upload_to_aws("tmp/image.jpg", f"{id}-raw.jpg")
  s3.upload_to_aws("tmp/circles.jpg", f"{id}-circles.jpg")
  s3.upload_to_aws("tmp/lines.jpg", f"{id}-lines.jpg")

  return {
    'body': {
      'raw': f"https://{BUCKET}.s3.eu-west-2.amazonaws.com/{id}-raw.jpg",
      'circles': f"https://{BUCKET}.s3.eu-west-2.amazonaws.com/{id}-circles.jpg",
      'lines': f"https://{BUCKET}.s3.eu-west-2.amazonaws.com/{id}-lines.jpg"  
    }
  }


@app.route('/hc', methods=['GET'])
def health_check():
  return {
    "body": "ok"
  }