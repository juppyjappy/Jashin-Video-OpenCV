import cv2
import numpy as np
from PIL import Image
import sys
import time
import random

#cascade 
face_cascade = "./haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(face_cascade)
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

device_id = 0
window_name = "Jashin Video"
mask1 = 'jashin2.png' #open
mask2 = 'jashin5.png' #one-closed
mask3 = 'jashin6.png' #both-closed

#画像の読み込み元
both = []
one = []
close = []
for i in range(6):
    both.append(Image.open('./open_jashin/ja'+str(i)+'.png'))
    one.append(Image.open('./one_jashin/ja'+str(i)+'.png'))
    close.append(Image.open('./closed_jashin/ja'+str(i)+'.png'))

def main(): 
    cv2.namedWindow(window_name)
    
    mask_pil = Image.open(mask3)
    mask_size = mask_pil.size
    img_back = Image.open(mask1)
    #img_back = cv2.cvtColor(img_back, cv2.COLOR_RGB2BGR)
    
    #時間管理変数
    #クールタイム用
    bef = -1.1
    bef2 = -1.1
    bef3 = -1.1
    eye_p = 0

    #傾き
    ang = 2

    cap = cv2.VideoCapture(device_id)
    ret, img = cap.read()
    img_back = img
    if ret:
        img_size = Image.fromarray(img).size
        mask_layer_origin = Image.new('RGBA', img_size, (255, 255, 255, 0))
 
    while ret:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(50, 50))

        """"face"""

        if len(face_list):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.convert('RGBA')
 
            mask_layer = mask_layer_origin.copy()

            """eye"""
            for x, y, w, h in face_list:
                face_gray = img_gray[y: y + h, x: x + w]

                eyes = eye_cascade.detectMultiScale(
                    face_gray,
                    scaleFactor=1.5,
                    minNeighbors=3,
                    minSize=(15, 15)
                )
            #目の決定
            now = time.time()
            if len(eyes) == 0 and now > bef + 0.5:
                eye_p = 2
                bef = time.time()
            elif len(eyes) == 1 and now > bef + 0.5:
                eye_p = 1
                bef = time.time()
            elif len(eyes) >= 2 and now > bef+0.5:
                eye_p = 0
                bef = time.time()
            #顔の傾き
            for (x, y, w, h) in face_list:
                if now >= bef3 + 0.7:
                    rand = random.random()
                    if rand <= 0.4:
                        ang = min(ang+1,5)
                    elif rand >= 0.6:
                        ang = max(ang-1,0)
                    bef3 = time.time()
                if eye_p == 0:
                    mask_pil = both[ang]
                elif eye_p == 1:
                    mask_pil = one[ang]
                else:
                    mask_pil = close[ang]
                mask_size = mask_pil.size
                #大きさと座標
                if now >= bef2 + 0.3:
                    scale = max(150,float(w))*2.1 / float(mask_size[0])
                    bef2 = time.time()
                mask_resize = (int(mask_size[0] * scale), int(mask_size[1] * scale))
                resized_mask_pil = mask_pil.resize(mask_resize, resample=Image.BILINEAR)
                

                #合成
                mask_layer.paste(resized_mask_pil, (max(0,x-int(scale*100)), max(0,y-int(scale*140))), resized_mask_pil)
                composite_pil = Image.alpha_composite(img_pil, mask_layer)



            img = np.asarray(composite_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img_back = img
        else:
            #映らなかった場合は前の画像を継続
            img = img_back
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
 
        cv2.imshow(window_name, img)
 
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
 
        ret, img = cap.read()
 
    cap.release() 
    cv2.destroyAllWindows()
 
 
if __name__ == '__main__':
    main()