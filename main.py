import cv2
import numpy as np
import glob

def cv_size(img):
  return tuple(img.shape[1::-1])

def detect_circles(img):

  # filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  # sharpen = cv2.filter2D(img, -1, filter)

  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  blur = cv2.medianBlur(imgGray, 5)
  # blur = cv2.GaussianBlur(img,(5,5),0)

  # bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

  edges = cv2.Canny(blur, 75, 140)


  circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 22, param1=8, param2=27,
                              minRadius=30, maxRadius=53)

  draw_circles_on_image(edges, circles)

  return edges


  # cv2.imshow('circles', img)

  # rho, theta, thresh = 2, np.pi/180, 400
  # lines = cv2.HoughLines(bin_img, rho, theta, thresh)

  # cv2.imshow("lines", lines)

  # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
  # imgCanny = cv2.Canny(imgGray, 100, 200)
  # cv2.imshow("imgGray", imgGray)
  # cv2.imshow("imgCanny", imgCanny)
  # return blur


def detect_lines(img):
  filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  sharpen = cv2.filter2D(img, -1, filter)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.blur(gray, (3, 3))

  # edges = cv2.Canny(blur, 100, 200)
  # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
  # for line in lines:
  #     x1, y1, x2, y2 = line[0]
  #     cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
  return blur

def draw_circles_on_image(img, circles):
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 3)


for image_name in glob.glob("images/*.jpg"):
  print(image_name.split('/')[1])
  img = cv2.imread(image_name, cv2.IMREAD_COLOR)
  circles = detect_circles(img)
  cv2.imwrite(f"circle-results/{image_name.split('/')[1]}", circles)

  lines = detect_lines(img)
  cv2.imwrite(f"line-results/{image_name.split('/')[1]}", lines)




# cv2.imshow("result", get_conturs(img))
# cv2.imshow("passed", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()