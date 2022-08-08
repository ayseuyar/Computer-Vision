import cv2
import numpy as np
from keras.models import load_model
from keras.datasets import cifar10
#loads the CIFAR dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


img_row, img_height, img_depth = 32,32,3

classifier = load_model("keras_cifar10_trained_model.h5")


color=True
scale = 8

def draw_test(name, res, input_im, scale, img_row, img_height):
    BLACK = [0,0,0]
    res = int(res)
    if res == 0:
        pred = "ucak"
    if res == 1:
        pred = "otomobil"
    if res == 2:
        pred = "kus"
    if res == 3:
        pred = "kedi"
    if res == 4:
        pred = "geyik"
    if res == 5:
        pred = "kopek"
    if res == 6:
        pred = "kurbaga"
    if res == 7:
        pred = "at"
    if res == 8:
        pred = "gemi"
    if res == 9:
        pred = "kamyon"

    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, input_im.shape[0]*2 ,cv2.BORDER_CONSTANT, value=BLACK)
    if color == False:
        expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (300,80) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)

    
for i in range(0,10):
    rand = np.random.randint(0, len(x_test))
    input_im = x_test[rand]
    imageL = cv2.resize(input_im, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    input_im = input_im.reshape(1, img_row, img_height, img_depth)

    ##get perdiction
    #res = str(classifier.predict_classes(input_im, 1, verbose = 0)[0])
    res = np.argmax(classifier.predict(input_im))

    draw_test("Lutfen Entera Basiniz", res, imageL, scale, img_row, img_height)
    cv2.waitKey(0)
cv2.destroyAllWindows()
 
