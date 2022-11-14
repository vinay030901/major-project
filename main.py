import cv2
import face_recognition

imgsrk = face_recognition.load_image_file('srk.jpg')
imgsrk = cv2.cvtColor(imgsrk, cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('srk_test.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

# ----------Finding face Location for drawing bounding boxes-------
facesrk = face_recognition.face_locations(imgsrk)[0]
encodesrk = face_recognition.face_encodings(imgsrk)[0]

# -------------------Drawing the Rectangle-------------------------
copy = imgsrk.copy()
cv2.rectangle(copy, (facesrk[3], facesrk[0]), (facesrk[1], facesrk[2]), (255, 0, 255), 2)

# for encoding image
facetest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
copytest = imgtest.copy()
cv2.rectangle(copytest, (facetest[3], facetest[0]), (facetest[1], facetest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodesrk], encodetest)
facedis = face_recognition.face_distance([encodesrk], encodetest)
print(facedis)
print(results)
cv2.putText(imgtest, f'{results} {round(facedis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('copy', copy)
cv2.imshow('elon', copytest)
cv2.waitKey(0)
