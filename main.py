import cv2
import numpy as np
import glob
# from dotenv import load_dotenv
# load_dotenv()

import src.recognition as rec

import src.s3 as s3

# s3.upload_to_aws("images/test1.jpg", "testfile.jpg")
# image_name = "images/test3.jpg"
# img = cv2.imread(image_name, cv2.IMREAD_COLOR)

# lines = rec.detect_lines_and_intersections(img)
# cv2.imwrite(f"line-results/{image_name.split('/')[1]}", lines)


for image_name in glob.glob("images/*.jpg"):
  print(image_name.split('/')[1])
  img = cv2.imread(image_name, cv2.IMREAD_COLOR)

  circles = rec.detect_circles(img)
  cv2.imwrite(f"circle-results/{image_name.split('/')[1]}", circles)

  lines = rec.detect_lines_and_intersections(img)
  cv2.imwrite(f"line-results/{image_name.split('/')[1]}", lines)




# cv2.imshow("result", get_conturs(img))
# cv2.imshow("passed", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()