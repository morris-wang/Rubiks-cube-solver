import cv2
import numpy as np
import math
from solvecube import utils

import serial
from time import sleep
import sys

STICKER_AREA_TILE_SIZE = 30
STICKER_AREA_TILE_GAP = 4
STICKER_AREA_OFFSET = 20

result_state = {}

snapshot_state = [
                  (255,255,255), (255,255,255), (255,255,255),
                  (255,255,255), (255,255,255), (255,255,255),
                  (255,255,255), (255,255,255), (255,255,255)]

preview_state  = [(255,255,255), (255,255,255), (255,255,255),
                (255,255,255), (255,255,255), (255,255,255),
                (255,255,255), (255,255,255), (255,255,255)]
state=  {
            'up':['w','w','w','w','y','w','w','w','w',],
            'left':['w','w','w','w','b','w','w','w','w',],
            'front':['w','w','w','w','r','w','w','w','w',],
            'right':['w','w','w','w','g','w','w','w','w',],
            'back':['w','w','w','w','o','w','w','w','w',],
            'down':['w','w','w','w','w','w','w','w','w',]
        }

color = {
        'y' : (0,255,255),
        'b'   : (255,0,0),
        'r'    : (0,0,255),
        'g'  : (0,255,0),
        'o' : (0,165,255),
        'w'  : (255,255,255)
        }

stickers = {
        'main': [
            [200, 120], [300, 120], [400, 120],
            [200, 220], [300, 220], [400, 220],
            [200, 320], [300, 320], [400, 320]
        ],
        'current': [
            [20, 20], [54, 20], [88, 20],
            [20, 54], [54, 54], [88, 54],
            [20, 88], [54, 88], [88, 88]
        ],
        'preview': [
            [20, 130], [54, 130], [88, 130],
            [20, 164], [54, 164], [88, 164],
            [20, 198], [54, 198], [88, 198]
        ],
        'left': [
            [50, 280], [94, 280], [138, 280],
            [50, 324], [94, 324], [138, 324],
            [50, 368], [94, 368], [138, 368]
        ],
        'front': [
            [188, 280], [232, 280], [276, 280],
            [188, 324], [232, 324], [276, 324],
            [188, 368], [232, 368], [276, 368]
        ],
        'right': [
            [326, 280], [370, 280], [414, 280],
            [326, 324], [370, 324], [414, 324],
            [326, 368], [370, 368], [414, 368]
        ],
        'up': [
            [188, 128], [232, 128], [276, 128],
            [188, 172], [232, 172], [276, 172],
            [188, 216], [232, 216], [276, 216]
        ],
        'down': [
            [188, 434], [232, 434], [276, 434],
            [188, 478], [232, 478], [276, 478],
            [188, 522], [232, 522], [276, 522]
        ], 
        'back': [
            [464, 280], [508, 280], [552, 280],
            [464, 324], [508, 324], [552, 324],
            [464, 368], [508, 368], [552, 368]
        ],
           }

average_sticker_colors = {}

def draw_preview_stickers(frame,stickers):
        stick=['front','back','left','right','up','down']
        for name in stick:
            for x,y in stickers[name]:
                cv2.rectangle(frame, (x,y), (x+20, y+20), (255,255,255), 2)

def fill_stickers(frame,stickers,sides):    
    for side,colors in sides.items():
        num=0
        for x,y in stickers[side]:
            cv2.rectangle(frame,(x,y),(x+40,y+40),color[colors[num]],-1)
            num+=1

colors = {  # momo's cube t20
    'r': ([150, 130, 210], [200, 180, 255]),
    'o': ([0, 100, 200], [20, 255, 255]),
    'b': ([80, 160, 180], [140, 240, 255]),
    'y': ([10, 100, 160], [60, 250, 240]),
    'w': ([5, 1, 180], [140, 50, 250]),
    'g': ([40, 120, 170], [110, 190, 250])
}

def find_contours(dilatedFrame):
    """Find the contours of a 3x3x3 cube."""
    contours = cv2.findContours(dilatedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    final_contours = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
        if len (approx) == 4: # is this is a rect
            area = cv2.contourArea(contour)
            (x, y, w, h) = cv2.boundingRect(approx)

 
            ratio = w / float(h)


            if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 60 and area / (w * h) > 0.4:
                final_contours.append((x, y, w, h))


    if len(final_contours) < 9:
        return []


    found = False
    contour_neighbors = {}
    for index, contour in enumerate(final_contours):
        (x, y, w, h) = contour
        contour_neighbors[index] = []
        center_x = x + w / 2
        center_y = y + h / 2
        radius = 1.5

    
        neighbor_positions = [
            # top left
            [(center_x - w * radius), (center_y - h * radius)],

            # top middle
            [center_x, (center_y - h * radius)],

            # top right
            [(center_x + w * radius), (center_y - h * radius)],

            # middle left
            [(center_x - w * radius), center_y],

            # center
            [center_x, center_y],

            # middle right
            [(center_x + w * radius), center_y],

            # bottom left
            [(center_x - w * radius), (center_y + h * radius)],

            # bottom middle
            [center_x, (center_y + h * radius)],

            # bottom right
            [(center_x + w * radius), (center_y + h * radius)],
        ]

        for neighbor in final_contours:
            (x2, y2, w2, h2) = neighbor
            for (x3, y3) in neighbor_positions:
               
                if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                    contour_neighbors[index].append(neighbor)

    for (contour, neighbors) in contour_neighbors.items():
        if len(neighbors) == 9:
            found = True
            final_contours = neighbors
            break

    if not found:
        return []

    
    y_sorted = sorted(final_contours, key=lambda item: item[1])

    
    top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
    middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
    bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

    sorted_contours = top_row + middle_row + bottom_row
    return sorted_contours


def detect(img, k):
    global current_color_to_calibrate_index
    global start_detect
    global text_position

    # if k == 122:  # if input == "z"
    #     reset_calibrate_mode()
    #     global calibrate_mode
    #     calibrate_mode = not calibrate_mode

    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurrFrame = cv2.blur(grayFrame, (3, 3))
    cannyFrame = cv2.Canny(blurrFrame, 30, 60, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilatedFrame = cv2.dilate(cannyFrame, kernel)

    contours = find_contours(dilatedFrame)
    if len(contours) == 9:
        draw_contours(img, contours)
        if start_detect == True:
            update_preview_state(img, contours)
        elif k == 115 and done_calibrating == False: # input s

            # print(colors_to_calibrate[current_color_to_calibrate_index] + " calibrate success")
            
            cv2.putText(preview, colors_to_calibrate[current_color_to_calibrate_index] + " calibrate success", (10, text_position), cv2.FONT_HERSHEY_TRIPLEX, 0.5, ( 209, 206, 0), 1, cv2.LINE_AA)
            text_position += 20

            current_color = colors_to_calibrate[current_color_to_calibrate_index]
            (x, y, w, h) = contours[4]
            roi = img[y+7:y+h-7, x+14:x+w-14]
            avg_bgr = get_dominant_color(roi)
            calibrated_colors[current_color] = avg_bgr
            current_color_to_calibrate_index += 1
            if current_color_to_calibrate_index == 6:
                set_cube_color_pallete(calibrated_colors)
                start_detect = True


def update_preview_state(frame, contours):

    for index, (x, y, w, h) in enumerate(contours):
        if index in average_sticker_colors and len(average_sticker_colors[index]) == 8:
            sorted_items = {}
            for bgr in average_sticker_colors[index]:
                key = str(bgr)
                if key in sorted_items:
                    sorted_items[key] += 1
                else:
                    sorted_items[key] = 1
            most_common_color = max(sorted_items, key=lambda i: sorted_items[i])
            average_sticker_colors[index] = []
            preview_state[index] = eval(most_common_color)
            break

        roi = frame[y+7:y+h-7, x+14:x+w-14]
        avg_bgr = get_dominant_color(roi)
        closest_color = get_closest_color(avg_bgr)['color_bgr']
        preview_state[index] = closest_color
        # print(preview_state)
        if index in average_sticker_colors:
            average_sticker_colors[index].append(closest_color)
        else:
            average_sticker_colors[index] = [closest_color]

def get_closest_color(bgr):
    """
    Get the closest color of a BGR color using CIEDE2000 distance.
    :param bgr tuple: The BGR color to use.
    :returns: dict
    """
    lab = bgr2lab(bgr)
    distances = []
    for color_name, color_bgr in cube_color_palette.items():
        distances.append({
            'color_name': color_name,
            'color_bgr': color_bgr,
            'distance': ciede2000(lab, bgr2lab(color_bgr))
        })
    closest = min(distances, key=lambda item: item['distance'])
    return closest

def bgr2lab(inputColor):
    """Convert BGR to LAB."""
    # Convert BGR to RGB
    inputColor = (inputColor[2], inputColor[1], inputColor[0])

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
         value = float(value) / 255

         if value > 0.04045:
              value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
         else:
              value = value / 12.92

         RGB[num] = value * 100
         num = num + 1

    XYZ = [0, 0, 0,]

    X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
    Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
    Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
    XYZ[ 0 ] = round( X, 4 )
    XYZ[ 1 ] = round( Y, 4 )
    XYZ[ 2 ] = round( Z, 4 )

    XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047            # ref_X =  95.047    Observer= 2°, Illuminant= D65
    XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0             # ref_Y = 100.000
    XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883          # ref_Z = 108.883

    num = 0
    for value in XYZ:

         if value > 0.008856:
              value = value ** ( 0.3333333333333333 )
         else :
              value = ( 7.787 * value ) + ( 16 / 116 )

         XYZ[num] = value
         num = num + 1

    Lab = [0, 0, 0]

    L = ( 116 * XYZ[ 1 ] ) - 16
    a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
    b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

    Lab [ 0 ] = round( L, 4 )
    Lab [ 1 ] = round( a, 4 )
    Lab [ 2 ] = round( b, 4 )

    return Lab

def ciede2000(Lab_1, Lab_2):
    """Calculates CIEDE2000 color distance between two CIE L*a*b* colors."""
    C_25_7 = 6103515625 # 25**7

    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)

    if b1_ == 0 and a1_ == 0: h1_ = 0
    elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
    else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi

    if b2_ == 0 and a2_ == 0: h2_ = 0
    elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
    else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0: dh_ = 0
    elif dh_ > math.pi: dh_ -= 2 * math.pi
    elif dh_ < -math.pi: dh_ += 2 * math.pi
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi
    else: h_ave = h1_ + h2_

    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)

    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0: h_ave_deg += 360
    elif h_ave_deg > 360: h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))

    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H

    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00


def draw_preview_stickers(frame):
    """Draw the current snapshot state onto the given frame."""
    # y = STICKER_AREA_TILE_SIZE * 3 + STICKER_AREA_TILE_GAP * 2 + STICKER_AREA_OFFSET * 2
    # draw_stickers(frame, preview_state, STICKER_AREA_OFFSET, y)
    # print(preview_state)
    draw_stickers(frame, preview_state, STICKER_AREA_OFFSET, STICKER_AREA_OFFSET)


def draw_snapshot_stickers(frame):
    """Draw the current snapshot state onto the given frame."""
    y = STICKER_AREA_TILE_SIZE * 3 + STICKER_AREA_TILE_GAP * 2 + STICKER_AREA_OFFSET * 2
    draw_stickers(frame, snapshot_state, STICKER_AREA_OFFSET, y)

def draw_stickers(frame, stickers, offset_x, offset_y):
    """Draws the given stickers onto the given frame."""
    index = -1
    for row in range(3):
        for col in range(3):
            index += 1
            x1 = (offset_x + STICKER_AREA_TILE_SIZE * col) + STICKER_AREA_TILE_GAP * col
            y1 = (offset_y + STICKER_AREA_TILE_SIZE * row) + STICKER_AREA_TILE_GAP * row
            x2 = x1 + STICKER_AREA_TILE_SIZE
            y2 = y1 + STICKER_AREA_TILE_SIZE

            # frame = np.array(frame)
            # shadow
            # print(type(frame))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                -1
            )

            # foreground color
            # print(stickers)
            
            cv2.rectangle(
                frame,
                (x1 + 1, y1 + 1),
                (x2 - 1, y2 - 1),
                get_prominent_color(stickers[index]),
                -1
            )

def get_prominent_color(bgr):
    global cube_color_palette
    """Get the prominent color equivalent of the given bgr color."""
    for color_name, color_bgr in cube_color_palette.items():
        
        if tuple([int(c) for c in bgr]) == color_bgr:
            # print(cube_color_palette[color_name])
            return cube_color_palette[color_name]
    return (0, 0, 0)

def set_cube_color_pallete(palette):
    """
    Set a new cube color palette. The palette is being used when the user is
    scanning his cube in solve mode by matching the scanned colors against
    this palette.
    """
    global cube_color_palette
    global color
    for side, bgr in palette.items():
        cube_color_palette[side] = tuple([int(c) for c in bgr])
        color[side] = tuple([int(c) for c in bgr])

    # print(cube_color_palette)
    # print(color)
    print("calibrate successful!!")


def get_dominant_color(roi):
    """
    Get dominant color from a certain region of interest.
    :param roi: The image list.
    :returns: tuple
    """
    average = roi.mean(axis=0).mean(axis=0)
    pixels = np.float32(roi.reshape(-1, 3))

    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return tuple(dominant)

def draw_contours(frame, contours):

    for index, (x, y, w, h) in enumerate(contours):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)

def reset_calibrate_mode():
    """Reset calibrate mode variables."""
    global current_color_to_calibrate_index
    global calibrated_colors
    global done_calibrating

    calibrated_colors = {}
    current_color_to_calibrate_index = 0
    done_calibrating = False



check_state=[]

cap=cv2.VideoCapture(0)
cv2.namedWindow('frame')

colors_to_calibrate = ['y', 'b', 'r', 'g', 'o', 'w']
calibrate_mode = False
cube_color_palette = {}

calibrated_colors = {}
current_color_to_calibrate_index = 0
done_calibrating = False

start_detect = False

text_position = 70


if __name__=='__main__':

    preview = np.zeros((600,800,3), np.uint8)
    # img = cv2.imread('4.jpg')
    # preview = cv2.resize(img, (800, 600))
    
    # print(preview)
    text = "yellow => blue => red => green => orange => white !!"

    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(preview, "Enter s to start calibrate", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
    

    while True:
        hsv=[]
        current_state=[]
        ret,img=cap.read()
        

        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask = np.zeros(frame.shape, dtype=np.uint8)   

        # draw_stickers(img,stickers,'main')

        # draw_preview_stickers(stickers)

        # fill_stickers(preview,stickers,state)
        
        k = cv2.waitKey(5) & 0xFF
         
        
        detect(img, k)

        if start_detect == True:
            global temp_state
            

            # draw_current_language(frame)
            draw_preview_stickers(img)
            
            preview = np.zeros((600,800,3), np.uint8)
            
            fill_stickers(preview,stickers,state)
            cv2.putText(preview, "L", (40, 270), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "F", (178, 270), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "R", (316, 270), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "U", (178, 118), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "D", (178, 424), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "B", (454, 270), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "press 'a' to start solving", (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (50, 200, 255), 1, cv2.LINE_AA)
            # draw_snapshot_stickers(img)
            # draw_scanned_sides(frame)
            # draw_2d_cube_state(frame)

            revert = {v : k for k, v in color.items()}
            # print(revert)
            temp_state = []


            if k == 102: # input = f
                # print("f")
                for i in range(9):
                    temp_state.append(revert[preview_state[i]])
                # print(temp_state)
                state['front'] = temp_state
                fill_stickers(preview,stickers,state)
            elif k == 100: # input = d
                # print("d")
                for i in range(9):
                    temp_state.append(revert[preview_state[i]])
                # print(temp_state)
                state['down'] = temp_state
                fill_stickers(preview,stickers,state)
            elif k == 108: # input = l
                # print("l")
                for i in range(9):
                    temp_state.append(revert[preview_state[i]])
                # print(temp_state)
                state['left'] = temp_state
                fill_stickers(preview,stickers,state)
            elif k == 114: # input = r
                # print("r")
                for i in range(9):
                    temp_state.append(revert[preview_state[i]])
                # print(temp_state)
                state['right'] = temp_state
                fill_stickers(preview,stickers,state)
            elif k == 117: # input = u
                # print("u")
                for i in range(9):
                    temp_state.append(revert[preview_state[i]])
                # print(temp_state)
                state['up'] = temp_state
                fill_stickers(preview,stickers,state)
            elif k == 98: # input = b
                # print("b")
                for i in range(9):
                    temp_state.append(revert[preview_state[i]])
                # print(temp_state)
                state['back'] = temp_state
                fill_stickers(preview,stickers,state)
            elif k == 97:
                string = ""
                print(state)
                for key in state:
                    for i in range(9):
                        string += (state[key][i])
                print(string)

                solution = utils.solve(string, 'Kociemba')
                print(solution)

                for i in range(len(solution)):
                    if  solution[i] == "R\'":
                        solution[i]="r"
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
                    elif  solution[i] == 'R2':     # R2:q  L2:a  B2:z  U2:w  D2:s  F2:x
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
            elif k == 116: #input t
                COM_PORT = 'COM6'  # need to change the port name
                BAUD_RATES = 9600
                ser = serial.Serial(COM_PORT, BAUD_RATES)
                i = 0
                try:
                    while i<len(solution):
                        if i==0:
                            sleep(1.5)
                        choice = solution[i]
                        if choice == 'r':    #R
                            print('r')
                            ser.write(b'r')  # 訊息必須是位元組類型
                            sleep(1.5)              # 暫停0.5秒，再執行底下接收回應訊息的迴圈
                        elif choice == 'R':
                            print('R')
                            ser.write(b'R')
                            sleep(1.5)
                        elif choice == 'q':
                            print('q')
                            ser.write(b'q')
                            sleep(1.5)
                        elif choice == 'L':     #L
                            print('L')
                            ser.write(b'L')
                            sleep(1.5)
                        elif choice == 'l':
                            print('l')
                            ser.write(b'l')
                            sleep(1.5)
                        elif choice == 'a':
                            print('a')
                            ser.write(b'a')
                            sleep(1.5)
                        elif choice == 'F':     #F
                            print('F')
                            ser.write(b'F')
                            sleep(1.5)
                        elif choice == 'f':
                            print('f')
                            ser.write(b'f')
                            sleep(1.5)
                        elif choice == 'x':
                            print('x')
                            ser.write(b'x')
                            sleep(1.5)
                        elif choice == 'B':     #B    
                            print('B')
                            ser.write(b'B')
                            sleep(1.5)
                        elif choice == 'b':
                            print('b')
                            ser.write(b'b')
                            sleep(1.5)
                        elif choice == 'z':
                            print('z')
                            ser.write(b'z')
                            sleep(1.5)
                        elif choice == 'U':     #U    
                            print('U')
                            ser.write(b'U')
                            sleep(1.5)
                        elif choice == 'u':
                            print('u')
                            ser.write(b'u')
                            sleep(1.5)
                        elif choice == 'w':
                            print('w')
                            ser.write(b'w')
                            sleep(1.5)
                        elif choice == 'D':     #D    
                            print('D')
                            ser.write(b'D')
                            sleep(1.5)
                        elif choice == 'd':
                            print('d')
                            ser.write(b'd')
                            sleep(1.5)
                        elif choice == 's':
                            print('s')
                            ser.write(b's')
                            sleep(1.5)
                        else:
                            print('bye!!')
                            ser.close()
                            sys.exit()

                        # while ser.in_waiting:
                        #     mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
                        #     print('控制板回應：', mcu_feedback)
                        i = i + 1

                except KeyboardInterrupt:
                    ser.close()
                    print('再見！')
                

        if k == 27:
            break


        cv2.imshow('preview',preview)
        cv2.imshow('frame',img)

    cv2.destroyAllWindows()