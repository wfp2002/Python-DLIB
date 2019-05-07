# import the necessary packages
from imutils import face_utils
import dlib
import cv2
 

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # Obtendo nossa imagem a webCam e transformando-a preto e branco.
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Detectando as faces em preto e branco.
    rects = detector(gray, 0)
    
    # para cada face encontrada, encontre os pontos de interesse.
    for (i, rect) in enumerate(rects):
        # fa predo e eno transforme isso em um array do numpy.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # desenhe na imagem cada cordenada(x,y) que foi encontrado.
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # Mostre a imagem com os pontos de interesse.
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()