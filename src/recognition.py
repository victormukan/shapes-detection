import cv2
import numpy as np
from collections import defaultdict
import sys
import math  


def cv_size(img):
  return tuple(img.shape[1::-1])

def detect_circles(img):
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  blur = cv2.medianBlur(imgGray, 5)

  edges = cv2.Canny(blur, 75, 140)

  circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 22, param1=8, param2=27, minRadius=30, maxRadius=53)
  
  centers = []
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
      center = (i[0], i[1])
      centers.append(center)

      cv2.circle(img, center, 1, (0, 0, 255), 7)
      radius = i[2]
      cv2.circle(img, center, radius, (255, 0, 255), 3)
  
  if len(centers) == 2:
    print(distance_between_two_points(centers[0], centers[1]))
  return img


def distance_between_two_points(x1, x2):
  return math.sqrt(math.pow(int(x1[0]) - int(x2[0]), 2) + math.pow(int(x1[1]) - int(x2[1]), 2)) 

def detect_lines(img):
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  blur = cv2.medianBlur(imgGray, 5)
  edges = cv2.Canny(blur, 15, 40)
  lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)

  for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

  # print(lines)

  # for line in segmented[0]:
  #   print(line)


  # for line in segmented[0]:
  #   # x1 = line[0][0][0]
  #   # y1 = line[0][0][1]
  #   # x2 = line[0][0][2]
  #   # y2 = line[0][0][3]
  #   x1, y1, x2, y2 = line[0]
  #   print(x1, y1, x2, y2)
  #   cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

  # for line in segmented[1]:
  #   x1, y1, x2, y2 = line[0]
  #   cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 3)
  return img


def detect_lines_and_intersections(img):
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  blur = cv2.medianBlur(imgGray, 5)
  # edges = cv2.Canny(blur, 20, 45)

  adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
  thresh_type = cv2.THRESH_BINARY_INV
  bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
  
  lines = cv2.HoughLines(bin_img, 1, np.pi/180, 380)

  # Cluster line angles into 2 groups (vertical and horizontal)
  segmented = segment_by_angle_kmeans(lines, 2)

  intersections = segmented_intersections(segmented)

  img_with_segmented_lines = np.copy(img)

  vertical_lines = segmented[1]
  drawLines(img_with_segmented_lines, vertical_lines, (0,255,0))

  horizontal_lines = segmented[0]
  drawLines(img_with_segmented_lines, horizontal_lines, (0,255,255))

  for point in intersections:
    pt = (point[0][0], point[0][1])
    length = 5
    cv2.line(img_with_segmented_lines, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 2) # vertical line
    cv2.line(img_with_segmented_lines, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 2)

  return img_with_segmented_lines




def segment_by_angle_kmeans(lines, k=2, **kwargs):

  # Define criteria = (type, max_iter, epsilon)
  default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
  criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

  flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
  attempts = kwargs.get('attempts', 10)

  # Get angles in [0, pi] radians
  angles = np.array([line[0][1] for line in lines])

  # Multiply the angles by two and find coordinates of that angle on the Unit Circle
  pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

  # Run k-means
  if sys.version_info[0] == 2:
      # python 2.x
      ret, labels, centers = cv2.kmeans(pts, k, criteria, attempts, flags)
  else: 
      # python 3.x, syntax has changed.
      labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

  labels = labels.reshape(-1) # Transpose to row vector

  # Segment lines based on their label of 0 or 1
  segmented = defaultdict(list)
  for i, line in zip(range(len(lines)), lines):
      segmented[labels[i]].append(line)

  segmented = list(segmented.values())
  print("Segmented lines into two groups: %d, %d" % (len(segmented[0]), len(segmented[1])))

  return segmented


def intersection(line1, line2):
  rho1, theta1 = line1[0]
  rho2, theta2 = line2[0]
  A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
  b = np.array([[rho1], [rho2]])
  x0, y0 = np.linalg.solve(A, b)
  x0, y0 = int(np.round(x0)), int(np.round(y0))

  return [[x0, y0]]


def segmented_intersections(lines):
  intersections = []
  for i, group in enumerate(lines[:-1]):
    for next_group in lines[i+1:]:
      for line1 in group:
        for line2 in next_group:
          intersections.append(intersection(line1, line2)) 

  return intersections


def drawLines(img, lines, color=(0,0,255)):
  for line in lines:
    for rho,theta in line:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))
      cv2.line(img, (x1,y1), (x2,y2), color, 2)