import cv2
import mediapipe as mp
import math
import numpy as np
from selenium import webdriver
import time
from PIL import Image
from io import BytesIO
import threading
import python_weather
import asyncio
import os
import calendar

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#開網頁的
firefox_web=webdriver.FirefoxOptions()
url='https://www.google.com.tw'
driver=webdriver.Firefox(options=firefox_web)
#存放網頁截圖
screenshot_arr = None
update_lock = threading.Lock()

#日曆
tc = calendar.TextCalendar()

w, h = 1080, 720   #畫面視窗大小

#天氣
async def getweather():
  # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
  async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
    # fetch a weather forecast from a city
    weather =await client.get('Taipei')
    
    # returns the current day's forecast temperature (int)
    # print(weather.current.temperature)
    draw[5:35, 220:350] = [255, 255, 255, 0]    #清空
    cv2.putText(draw,'Taipei:'+str(round(((weather.current.temperature)-32)*5/9))+' C',(180,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0,255),2,cv2.LINE_AA)
    time.sleep(60)

# 函數：截取網頁截圖
def capture_screenshot():
    global screenshot_arr
    global update_lock

    while True:
        screenshot = driver.get_screenshot_as_png()

        with update_lock:
            screenshot_arr = np.array(Image.open(BytesIO(screenshot)))

        time.sleep(1)  # 等待1秒再截取下一張截圖

# 函數：顯示截圖
def show_screenshot(img):
    global screenshot_arr
    global update_lock

    while True:
        with update_lock:
            if screenshot_arr is not None:
                resized_screenshot = cv2.resize(screenshot_arr, (400, 200))
                img[50:50+resized_screenshot.shape[0], 100:100+resized_screenshot.shape[1]] = resized_screenshot

# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '1'
    elif f1<50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    else:
        return ''

cap = cv2.VideoCapture(0)            # 讀取攝影機
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv2.LINE_AA               # 印出文字的邊框
nowtime=None

# mediapipe 啟用偵測手掌
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()                                     # 影像尺寸
    draw = np.zeros((h,w,4), dtype='uint8')                # 繪製全黑背景，尺寸和影像相同
    # white=255 - np.zeros((h,w,4), dtype='uint8')
    dots = []                                              # 使用 dots 空串列記錄繪圖座標點
    cv2.circle(draw,(40,40),20,(255,0,0,255),-1)
    cv2.putText(draw,'web',(28,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0,255),1,cv2.LINE_AA)

    screenshot_thread_started = False
    screenshot_thread = threading.Thread(target=capture_screenshot)
    show_screenshot_thread = threading.Thread(target=show_screenshot, args=(draw,))

    #天氣
    asyncio.run(getweather())

    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (w,h))                       
        # img = cv2.flip(img, 1)
        camera_counter=0
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # 偵測手勢的影像轉換成 RGB 色彩
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)        # 畫圖的影像轉換成 BGRA 色彩
        results = hands.process(img2)                      # 偵測手勢
        #取得時間
        t=time.time()
        t1=time.localtime(t)
        nowtime=time.strftime('%Y/%m/%d %H:%M:%S',t1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []                         # 記錄手指節點座標的串列
                for i in hand_landmarks.landmark:
                    # 將 21 個節點換算成座標，記錄到 finger_points
                    x = i.x*w
                    y = i.y*h
                    finger_points.append((x,y))
                if finger_points:
                    finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
                    text = hand_pos(finger_angle)            # 取得手勢所回傳的內容
                    if text == '1':
                        fx = int(finger_points[8][0])        # 如果手勢為 1，記錄食指末端的座標
                        fy = int(finger_points[8][1])
                        if fy>=20 and fy<=60 and fx>=20 and fx<=60:
                            # 開始截圖和顯示截圖的執行緒
                            if not screenshot_thread_started:
                                screenshot_thread_started = True
                                driver.get(url)
                                screenshot_thread.start()
                                show_screenshot_thread.start()
                        #顯示日曆
                        elif fx<=15:
                            calendar_text=tc.formatmonth(t1.tm_year,t1.tm_mon)
                            calendar_draw=np.zeros((300,400,3), dtype=np.uint8)
                            y0, dy = 15, 50
                            for i, line in enumerate(calendar_text.split('\n')):
                                y = y0 + i * dy
                                cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        else:
                            dots.append([fx,fy])             # 記錄食指座標
                    if text == 'ok':                #儲存照片
                        if camera_counter == 0:
                            print('save')
                            camera_counter=1
                        else:
                            pass

                    else:
                        dots = [] # 如果換成別的手勢，清空 dots

        #把日期時間寫在圖上
        draw[5:35, w-190:w] = [255, 255, 255, 0]    #清空
        cv2.putText(draw,nowtime,(w-190,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0,255),2,cv2.LINE_AA)   #寫入
        
        # 將影像和黑色畫布合成
        for j in range(w):
            img[:,j,0] = img[:,j,0]*(1-draw[:,j,3]/255) + draw[:,j,0]*(draw[:,j,3]/255)
            img[:,j,1] = img[:,j,1]*(1-draw[:,j,3]/255) + draw[:,j,1]*(draw[:,j,3]/255)
            img[:,j,2] = img[:,j,2]*(1-draw[:,j,3]/255) + draw[:,j,2]*(draw[:,j,3]/255)
        
        cv2.imshow('output', img)

        keyboard = cv2.waitKey(5)
        if keyboard == ord('q'):
            if screenshot_thread and screenshot_thread.is_alive():
                screenshot_thread.join()
            if show_screenshot_thread and show_screenshot_thread.is_alive():
                show_screenshot_thread.join()
            driver.quit()
            break

cap.release()
cv2.destroyAllWindows()