import math

def distance_between_two_points(x1, x2):
  return math.sqrt(math.pow(int(x1[0]) - int(x2[0]), 2) + math.pow(int(x1[1]) - int(x2[1]), 2)) 

def find_lines_between_points(centers): 
  result = []
  i = 0
  while i < len(centers):
    j = i + 1
    while j < len(centers):
      result.append({ 
        "x": centers[i],
        "y": centers[j], 
        "length": distance_between_two_points(centers[i], centers[j])
      })
      j += 1
    i += 1

  return result