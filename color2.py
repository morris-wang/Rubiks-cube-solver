import cv2 as cv
import numpy as np
from imutils import contours
from solvecube import utils
# momo's cube t20
colors = {  
    'r': ([150, 130, 210], [200, 180, 255]),
    'o': ([0, 100, 200], [20, 255, 255]),
    'b': ([80, 160, 180], [140, 240, 255]),
    'y': ([10, 100, 160], [60, 250, 240]),
    'w': ([5, 1, 180], [140, 50, 250]),
    'g': ([40, 120, 170], [110, 190, 250])
}

a = [[0]*2000 for i in range(2000)]
b = [0]*9
k = 0


def detect(color, image):
    original = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = np.zeros(image.shape, dtype=np.uint8)
    open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    lower = np.array(colors[color][0], dtype=np.uint8)
    upper = np.array(colors[color][1], dtype=np.uint8)
    color_mask = cv.inRange(image, lower, upper)
    color_mask = cv.morphologyEx(
        color_mask, cv.MORPH_OPEN, open_kernel, iterations=18)
    color_mask = cv.morphologyEx(
        color_mask, cv.MORPH_CLOSE, close_kernel, iterations=5)
    color_mask = cv.merge([color_mask, color_mask, color_mask])
    mask = cv.bitwise_or(mask, color_mask)
    gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    cnts = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if cnts:
        (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
        cube_rows = []
        row = []
        for (i, c) in enumerate(cnts, 1):
            row.append(c)
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            cube_rows.append(cnts)
            row = []
        number = 0
        for row in cube_rows:
            for c in row:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(original, (x, y),
                              (x + w, y + h), (36, 255, 12), 2)
                number += 1
                if a[x][y] == 0:
                    a[x][y] = color


def sequence1(x, y, i1, j1):  # momo's cube t20
    for i in range(210):
        for j in range(210):
            if a[i1 + i + 250 * x - 130][j1 + j + 250 * y - 130] != 0:
                global k
                if k == 9:
                    k = 0
                b[k] = a[i1 + i + 250 * x - 130][j1 + j + 250 * y - 130]
                k = k+1


def together(image):
    detect('y', image)
    detect('g', image)
    detect('b', image)
    detect('o', image)
    detect('w', image)
    detect('r', image)

    global a
    min1 = 3000
    i1 = 0
    j1 = 0
    for i in range(2000):
        for j in range(2000):
            if a[i][j] != 0:
                if i + j < min1:
                    min1 = i + j
                    i1 = i
                    j1 = j

    sequence1(0, 0, i1, j1)
    sequence1(1, 0, i1, j1)
    sequence1(2, 0, i1, j1)
    sequence1(0, 1, i1, j1)
    sequence1(1, 1, i1, j1)
    sequence1(2, 1, i1, j1)
    sequence1(0, 2, i1, j1)
    sequence1(1, 2, i1, j1)
    sequence1(2, 2, i1, j1)

    for i in range(9):
        global counter, b
        c[counter] = b[i]
        counter = counter + 1
    # global a
    a = [[0]*2000 for i in range(2000)]
    # global b
    b = [0]*9


counter = 0
c = [0]*54

image1 = cv.imread('u1.jpg')
image2 = cv.imread('u2.jpg')
image3 = cv.imread('u3.jpg')
image4 = cv.imread('u4.jpg')
image5 = cv.imread('u5.jpg')
image6 = cv.imread('u6.jpg')

together(image1)
together(image2)
together(image3)
together(image4)
together(image5)
together(image6)

count = 0
for t in range(54):
    # print(c[t],end='')
    if c[t] != 0:
        count = count + 1
print(count)

if count == 54:
    string = "".join(c)
    print(string)

    solution = utils.solve(string, 'Kociemba')
    print(solution)

    for i in range(len(solution)):      # R2:q  L2:a  B2:z  U2:w  D2:s  F2:x 
        if  solution[i] == "R\'":
            solution[i]='r'
        elif  solution[i] == 'L\'':
            solution[i]='l'
        elif  solution[i] == 'B\'':
            solution[i]='b'
        elif  solution[i] == 'U\'':
            solution[i]='u'
        elif  solution[i] == 'D\'':
            solution[i]='d'
        elif  solution[i] == 'F\'':
            solution[i]='f'
        elif  solution[i] == 'R2':
            solution[i]='q'
        elif  solution[i] == 'L2':
            solution[i]='a'
        elif  solution[i] == 'B2':
            solution[i]='z'
        elif  solution[i] == 'U2':
            solution[i]='w'
        elif  solution[i] == 'D2':
            solution[i]='s'
        elif  solution[i] == 'F2':
            solution[i]='x'
    print(solution)


