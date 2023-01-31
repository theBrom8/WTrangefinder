# starting to learn how to bot for games
# by Johnny Jordi
#


#pip install pyautogui
#pip install pymsgbox
#pip install tk
#pip install opencv-contrib-python-headless
#more info:	https://pyautogui.readthedocs.io/en/latest/install.html

import pyautogui, sys, time, cv2, pymsgbox, Tkinter


#vars
targetImage = str(r'images\aimTrainer\target1.jpg');



#Helper functions

def click_Image(image, conf, action = "left"):
	x, y = pyautogui.locateCenterOnScreen(image, confidence= conf);
	pyautogui.click(x, y,button = action);

#ini
exit = False;

#Action starts here
print("Hello, this is a Bot made by Brom8. \n your screenresolution is: " + str(pyautogui.size()))

while not exit:
    i = 0
    while i < 10:
        #time.sleep(2)
        try:
            click_Image(targetImage,.5)
        except:
            print("failed to click: ")
        i += 1
    userAction = pymsgbox.confirm(text='wanna do 10 more?', title='10 targets done', buttons=['yes', 'no'])
    if userAction == "no":
        exit = True;
        print("\n user ended program")








