import cv2
import numpy as np
import glob
import src.recognition as rec
import src.s3 as s3


# image_name = 'images/test.jpg'
# print(image_name)
# img = cv2.imread(image_name, cv2.IMREAD_COLOR)

# circles = rec.detect_circles(img)
# cv2.imwrite(f"circle-results/{image_name.split('/')[1]}", circles['image'])

# lines = rec.detect_lines_and_intersections(img)
# cv2.imwrite(f"line-results/{image_name.split('/')[1]}", lines)


for image_name in glob.glob("images/*.jpg"):
  print(image_name.split('/')[1])
  img = cv2.imread(image_name, cv2.IMREAD_COLOR)

  circles = rec.detect_circles(img)
  cv2.imwrite(f"circle-results/{image_name.split('/')[1]}", circles['image'])

  lines = rec.detect_lines(img)
  cv2.imwrite(f"line-results/{image_name.split('/')[1]}", lines)
