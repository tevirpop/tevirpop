'''
import cv2

# Загрузите каскадный классификатор для распознавания лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Инициализировать объект видеозахвата
cap = cv2.VideoCapture(0)

while True:
    # Прочитать кадр с камеры
    ret, frame = cap.read()

    # Преобразовать кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Обнаружение лиц в рамке оттенков серого
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Нарисуйте прямоугольник вокруг каждой обнаруженной грани
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Для растягивания изображения
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    # Отобразить результирующий кадр
    cv2.imshow('frame', frame)

    # Выйдите из цикла, если нажата клавиша 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Отпустите объект видеозахвата и закройте окно
cap.release()
cv2.destroyAllWindows()
'''

# Импортируем необходимые библиотеки
import cv2
import mediapipe as mp
import math

#from google.protobuf.json_format import MessageToDict

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

stroka = ""
stroka2 = ''
stroka3 = ''
strokaVnizu = ""
strokaZnak = ''
stopsl = 0
stops = 0

# Создаем объекты для обнаружения и отрисовки рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Открываем видеопоток с веб-камеры
cap = cv2.VideoCapture(0)

# В цикле читаем кадры и обрабатываем их
while cap.isOpened():
    # Читаем кадр и преобразуем его в RGB
    success, image = cap.read()
    if not success:
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Передаем кадр в модель обнаружения рук
    results = hands.process(image)

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Если есть результаты, то отрисовываем их на кадре
    #faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    # Нарисуйте прямоугольник вокруг каждой обнаруженной грани
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #for hand_landmarks in results.multi_hand_landmarks:
        #    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_scale2 = 4
            font_scale3 = 3
            font_scale4 = 2
            font_scale5 = 0.7
            font_color = (255, 255, 255)
            font_color5 = (0, 0, 0)
            line_type = 1
            line_type2 = 4

            # Извлекаем координаты кончиков пальцев
            landmarks = [[lmk.x, lmk.y] for lmk in hand_landmarks.landmark]
            finger_tip_coords = [landmarks[8], landmarks[12], landmarks[16], landmarks[20], landmarks[4]]

            # Вычисляем расстояние между кончиками пальцев с помощью теоремы косинусов
            distances = []
            for i in range(len(finger_tip_coords)):
                for j in range(i + 1, len(finger_tip_coords)):
                    x1, y1 = finger_tip_coords[i]
                    x2, y2 = finger_tip_coords[j]
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    distances.append(distance)

            #for j in results.multi_handedness:
            #    label = MessageToDict(j)['classification'][0]['label']

                #if label == 'Left':
                #    cv2.putText(image, label + ' Hand', (20, 280), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                #if label == 'Right':
                #    cv2.putText(image, label + ' Hand', (460, 280), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

            if (handedness.classification[0].label == 'Left'):
                x1, y1 = landmarks[4]
                cv2.putText(image, f".", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[8]
                cv2.putText(image, f"6", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[12]
                cv2.putText(image, f"7", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[16]
                cv2.putText(image, f"8", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[20]
                cv2.putText(image, f"9", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[17]
                cv2.putText(image, f"0", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[5]
                cv2.putText(image, f"", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[6]
                cv2.putText(image, f"-", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)

            if (handedness.classification[0].label == 'Right'):
                x1, y1 = landmarks[4]
                cv2.putText(image, f".", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[8]
                cv2.putText(image, f"5", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[12]
                cv2.putText(image, f"4", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[16]
                cv2.putText(image, f"3", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[20]
                cv2.putText(image, f"2", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[17]
                cv2.putText(image, f"1", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[5]
                cv2.putText(image, f"<-", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)
                x1, y1 = landmarks[6]
                cv2.putText(image, f"+", (round(x1 * 635), round(y1 * 476)), font, font_scale5, font_color5, line_type2)

            #определяем растояние между кончиком большого пальца и основы мизинца 11
            x1, y1 = landmarks[4]
            x2, y2 = landmarks[17]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)

            # определяем растояние между кончиком большого пальца и указательный средина 12
            x1, y1 = landmarks[4]
            x2, y2 = landmarks[6]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)
            # определяем растояние между кончиком большого пальца и фак средина 13
            x1, y1 = landmarks[4]
            x2, y2 = landmarks[10]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)
            # определяем растояние между кончиком большого пальца и хз средина 14
            x1, y1 = landmarks[4]
            x2, y2 = landmarks[14]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)
            # определяем растояние между кончиком большого пальца и хз средина 15
            x1, y1 = landmarks[4]
            x2, y2 = landmarks[5]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)

            #landmarksn = [[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]
            #x1, y1, z1 = landmarksn[4]
            #x2, y2, z2 = landmarksn[17]
            #distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            #distances.append(distance)
            #cv2.putText(image, f"x: {landmarks[4]}", (10, 300), font, font_scale, font_color, line_type)

            # Выводим расстояния на экран


            friz = 20

            #cv2.putText(image, f".", (635, 475), font, font_scale, font_color, line_type)

            #Пишем подписи возле точек пальцев
            #for i in range(len(finger_tip_coords)):
            #    x1, y1 = finger_tip_coords[i]
            #    cv2.putText(image, f"{i+1}", (round(x1*635), round(y1*476)), font, font_scale, font_color, line_type)


            #for i in range(len(distances)):

                #strokaVnizu = '.'
                #if (handedness.classification[0].label == 'Left'):
                #    cv2.putText(image, f"Distance {i + 1}: {distances[i]:.2f} pixels", (00, (i + 1) * 20), font, font_scale,
                #            font_color, line_type)
                #if (handedness.classification[0].label == 'Right'):
                #    cv2.putText(image, f"Distance {i + 1}: {distances[i]:.2f} pixels", (450, (i + 1) * 20), font, font_scale,
                #            font_color, line_type)

            if (distances[10] < 0.05 and handedness.classification[0].label == 'Left'):
                strokaVnizu = '0'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
                #cv2.putText(image, stroka, (200, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[3] < 0.05 and handedness.classification[0].label == 'Left'):
                strokaVnizu = '6'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[6] < 0.05 and handedness.classification[0].label == 'Left'):
                strokaVnizu = '7'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[8] < 0.05 and handedness.classification[0].label == 'Left'):
                strokaVnizu = '8'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[9] < 0.05 and handedness.classification[0].label == 'Left'):
                strokaVnizu = '9'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[11] < 0.03 and handedness.classification[0].label == 'Left'):
                strokaZnak = '-'
                if (stops > friz):
                    stops = 1
                    if (stroka == ""):
                        stroka = '0'
                    if (stroka2 == ""):
                        stroka2 = '0'
                    if (stroka3 == ""):
                        stroka3 = '0'
                    stroka = int(stroka3) - int(stroka2)
                    stroka3 = stroka
                    stroka2 = ''
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            if (distances[3] < 0.05 and handedness.classification[0].label == 'Right'):
                strokaVnizu = '5'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[6] < 0.05 and handedness.classification[0].label == 'Right'):
                strokaVnizu = '4'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[8] < 0.05 and handedness.classification[0].label == 'Right'):
                strokaVnizu = '3'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[9] < 0.05 and handedness.classification[0].label == 'Right'):
                strokaVnizu = '2'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[10] < 0.05 and handedness.classification[0].label == 'Right'):
                strokaVnizu = '1'
                if (stops > friz):
                    stops = 1
                    stroka2 = str(stroka2) + str(strokaVnizu)
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            #elif (distances[11] < 0.015 and handedness.classification[0].label == 'Right'):
            #    strokaVnizu = '='
            #    if (stops == 0):
            #        stops = 1
            #        stroka2 = stroka
            #    cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            elif (distances[11] < 0.03 and handedness.classification[0].label == 'Right'):
                strokaZnak = "+"
                if (stops > friz):
                    stops = 1
                    if (stroka == ""):
                        stroka = '0'
                    if (stroka2 == ""):
                        stroka2 = '0'
                    if (stroka3 == ""):
                        stroka3 = '0'
                    stroka = int(stroka2) + int(stroka3)
                    stroka3 = stroka
                    stroka2 = ''
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)

            elif (distances[14] < 0.015 and handedness.classification[0].label == 'Right'):
                strokaVnizu = '<-'
                if (stops > 10):
                    stops = 1
                    stroka2 = str(stroka2)[:-1]
                cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
            else:
                #strokaVnizu = '.'
                stops += 1
                #cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
                cv2.putText(image, str(stroka), (100, image.shape[0] - 10), font, font_scale3, font_color, line_type2)
                cv2.putText(image, str(stroka2), (150, image.shape[0] - 110), font, font_scale4, font_color, line_type2)
                cv2.putText(image, str(stroka3), (150, image.shape[0] - 210), font, font_scale4, font_color, line_type2)
                cv2.putText(image, str(strokaZnak), (100, image.shape[0] - 160), font, font_scale4, font_color, line_type2)


               # if (distances[10] < 0.05 and handedness.classification[0].label == 'Right'):
               #     strokaVnizu = '*'
               #     cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)
               # if (distances[10] < 0.05 and handedness.classification[0].label == 'Right'):
               #     strokaVnizu = '/'
               #     cv2.putText(image, strokaVnizu, (10, image.shape[0] - 10), font, font_scale2, font_color, line_type2)

    # Возвращаем цветовое пространство в BGR и показываем кадр на экране
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Добавляем строку в нижнюю часть экрана
    #cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), font, font_scale, font_color, line_type)
    #Зеркалим
    #flipped_image = cv2.flip(image, 1)
    cv2.imshow('Hand Tracking', image)

    # Если нажата клавиша Q, то выходим из цикла
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
cv2.destroyAllWindows()