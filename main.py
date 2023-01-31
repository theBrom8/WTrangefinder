# tests with openCV
# will try programm wo WT
# by Johnny Jordi



import cv2, pyautogui, time, keyboard, numpy, pymsgbox, PIL, pytesseract, os, math
#pip install pytesseract - used for string interpretation
# and install tesseract - https://github.com/UB-Mannheim/tesseract/wiki
# cv2 - openCV, computervision - pip install opencv-python
# pyautogiu - controls for computer
# numpy is needed for a lot of cv2 things
# PIL - install pillow


#libary things
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#def

def test_threshold(img, path, name):
        i = 254 # test all threshholds
        while i >= 0:
            th = cv2.threshold(img,i,255,cv2.THRESH_BINARY)[1]
            cv2.imwrite(path + "\\" + name + str(i) + ".png", th)
            i -= 1 


def get_map(area=(0, 0, 1920, 1080)):
    # make screenshot
    img = pyautogui.screenshot("img.png", region = area)
    #convert to cv2
    img =  numpy.array(img)
    #convert from RGB to BGR plane
    return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

def get_map_safe(area=(0, 0, 1920, 1080)):
    # make screenshot
    img = pyautogui.screenshot("img.png", region = area)
    #convert to cv2
    img =  numpy.array(img)
    #convert from RGB to BGR plane
    return cv2.imread("img.png")

def first_int_from_string(s):
    number = "0"
    for element in str(s):
        if (element >= '0' and element <= '9'):
            number += (element)
        else:
            return int(number)

def calculate_meter_per_pixel(img): # will need an image of the scale part of the map
    
    #prepare image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thNumbers = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)[1]

    #read meters
    sMeters =  pytesseract.image_to_string(thNumbers, config="--psm 7")
    meters = first_int_from_string(sMeters)
    if meters == None or (meters < 50):
        meters = int(pymsgbox.prompt("Brom Ballistic Calculators couldn't read the scale, pleas enter in Meters:", default="250"))

    ## pixellenght of the line
    # lets only look at the bottom 25 pixels
    y, x = gray.shape
    lineImg = gray#[25:y, 0:x] #im[y1:y2, x1:x2]

    thLine = cv2.threshold(lineImg,10,255,cv2.THRESH_BINARY)[1]

    #invert image
    invLine = numpy.invert(thLine)

    if debugmode:
        cv2.imwrite("line.png", lineImg)
        cv2.imwrite("thLine.png", thLine)
        cv2.imwrite("invLine.png", invLine)
        cv2.imwrite("thNumbers.png", thNumbers)

    # get contours CHAIN_APPROX_NONE  or  CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(image=invLine, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    with_contours = cv2.drawContours(img, contours, -1,(255,0,255),-1)
    if debugmode:
        cv2.imshow('Detected contours', with_contours)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    scaleLength = 1

    # Draw a bounding box around all contours
    for c in contours:
            # Make sure contour area is large enough
        area = cv2.contourArea(c)

        if area > 10:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,0,0), 5)
            if w > scaleLength:
                scaleLength = w

    if debugmode:        
        cv2.imshow('All contours with bounding box', with_contours)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    if scaleLength > (100*ratio):
        scaleLength = 100 * ratio

    return float(meters/scaleLength)


def match_Contour(playerImg, mapImg, filterYellow = False, debugColor = (255,255,0)):

    #use threshold on map, 160 was the best by experimental Tests
    if filterYellow:
        #move to hsv
        hsv = cv2.cvtColor(mapImg, cv2.COLOR_BGR2HSV)
      
        # Threshold of blue in HSV space
        lower_yellow = numpy.array([20, 100, 100])
        upper_yellow = numpy.array([40, 255, 255])
    
        # override Image
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        if debugmode:
            cv2.imwrite(r"mask.png", mask)
        mapImg =  res = cv2.bitwise_and(mapImg,mapImg, mask= mask)

    grayMap = cv2.cvtColor(mapImg, cv2.COLOR_BGR2GRAY)
    thMap = cv2.threshold(grayMap,160,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite(r"images\WT\thMap.png", thMap)
    mapContours, _ = cv2.findContours(thMap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get the playersshape
    grayPlayer = cv2.cvtColor(playerImg, cv2.COLOR_BGR2GRAY)
    thPlayer = cv2.threshold(grayPlayer,150,255,cv2.THRESH_BINARY)[1]
    cv2.imwrite(r"images\WT\thPlayer.png", thPlayer)
    playerContours, _ = cv2.findContours(thPlayer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #matching
    bestMatchResult = 1.0
    bestMatchIndex = 0
    for i,c in enumerate(mapContours):
        #match all shapes
        match = cv2.matchShapes(playerContours[0],c, 1, 0.0)

        #keep the best, aka smalest distance
        if match < bestMatchResult:
            bestMatchResult = match
            bestMatchIndex = i
    
    #calculate center of best match
    M = cv2.moments(mapContours[bestMatchIndex])
    # Movements contain diffrent properties, the centroid is found like this:
    playerX = int(M["m10"] / M["m00"])
    playerY = int(M["m01"] / M["m00"])

    if debugmode:
        cv2.rectangle(mapImg,(playerX-5,playerY-5), (playerX+5,playerY+5), debugColor, 5)
        cv2.imshow('All contours with bounding box', mapImg)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    return playerX, playerY


def match_template(mapImg, pingImg):
    # All the 6 methods for comparison in a list (0-5)
    method = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    iconY, iconX, z = pingImg.shape

    #match
    res = cv2.matchTemplate(mapImg,pingImg,5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #pingPos = max_loc + (iconX /2, iconY / 2)
    pingPos = min_loc + (iconX /2, iconY / 2)

    if debugmode:
        cv2.rectangle(mapImg,(pingPos[0]-5,pingPos[1]-5), (pingPos[0]+5,pingPos[1]+5), (255,0,255), 5)
        cv2.imshow('All contours with bounding box', mapImg)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    return pingPos[0], pingPos[1]




# ini vars
global debugmode
debugmode = True
meterPx = 0
ratio = 1.0
mapArea = (3180*ratio,1500*ratio, 650*ratio, 650*ratio)
#scaleArea = (500*ratio,610*ratio, 140*ratio, 40*ratio)

# main loop
#map = cv2.imread(r"images\WT\wholeMap.jpg")
player = cv2.imread(r"images\WT\player.png")
#scale = cv2.imread(r"images\WT\scale.jpg")
ping = cv2.imread(r"images\WT\ping.png")


while True:
    time.sleep(0.1)
    #print(cv2.cvtColor(numpy.uint8([[[0,255,255 ]]]),cv2.COLOR_BGR2HSV)) #print yellow in hsv


    #x = get_matched_coordinates(image, player)
    if keyboard.is_pressed('Home'):
        map = get_map_safe(area=(mapArea))
        scale = map[int(610*ratio):int(645*ratio), int(500*ratio):int(640*ratio)] #im[y1:y2, x1:x2]
        meterPx = calculate_meter_per_pixel(scale)
        time.sleep(0.1)

    if keyboard.is_pressed('§'):

        # try:

        map = get_map_safe(area=(mapArea))

        if meterPx ==0:
            scale = map[int(610*ratio):int(645*ratio), int(500*ratio):int(640*ratio)] #im[y1:y2, x1:x2]
            meterPx = calculate_meter_per_pixel(scale)
        

        xPlayer, yPlayer = match_Contour(player, map)
        #xPing, yPing = find_pings(map, ping)
        xPing, yPing = match_Contour(ping, map, filterYellow=True, debugColor=(0,255,0))
        
        distancePixel = math.sqrt(math.pow(xPing-xPlayer,2)+math.pow(yPing-yPlayer,2))

        pymsgbox.alert(title="Distance", text=str(distancePixel*meterPx), timeout=800)
        time.sleep(0.1)

        # except Exception as e:
        #     pymsgbox.alert(title="Error", text=e, timeout=20000)


