import cv2
from flask import Flask, request, Response
import os
import src.recognition as rec
import src.s3 as s3
from uuid import uuid4

BUCKET = os.environ.get("AWS_BUCKET")


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
  if 'file' not in request.files:
    return Response(
        "Missing image",
        status=400,
    )
  
  file = request.files['file']
  file.save(os.path.join('./tmp/', "image.jpg"))

  img = cv2.imread("tmp/image.jpg", cv2.IMREAD_COLOR)

  id = str(uuid4())

  circles = rec.detect_circles(img)
  cv2.imwrite("tmp/circles.jpg", circles['image'])

  
  s3.upload_to_aws("tmp/image.jpg", f"{id}-raw.jpg")
  s3.upload_to_aws("tmp/circles.jpg", f"{id}-circles.jpg")

  show_intermediate = request.args.get('show_intermediate')

  intermediate = []
  if show_intermediate: 
    for i, int_img in enumerate(circles['intermediateImages']):
      cv2.imwrite(f"tmp/intermediate-{i}.jpg", int_img)
      s3.upload_to_aws(f"tmp/intermediate-{i}.jpg", f"{id}-intermediate-{i}.jpg")
      intermediate.append(f"https://{BUCKET}.s3.eu-west-2.amazonaws.com/{id}-intermediate-{i}.jpg")


  return {
    'body': {
      'raw': f"https://{BUCKET}.s3.eu-west-2.amazonaws.com/{id}-raw.jpg",
      'circles': f"https://{BUCKET}.s3.eu-west-2.amazonaws.com/{id}-circles.jpg",
      'lines': f"https://{BUCKET}.s3.eu-west-2.amazonaws.com/{id}-lines.jpg",
      'maxCenterDistance': circles['maxCenterDistance'],
      'circlesCount': circles['circlesCount'],
      'intermediate': intermediate,
      'isIncorrect': f"{circles['maxCenterDistance'] > 0.1}",
      'acceptableDistance': 0.1
    }
  }


@app.route('/hc', methods=['GET'])
def health_check():
  return {
    "body": "ok"
  }