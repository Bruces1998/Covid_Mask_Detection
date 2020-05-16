from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from DLWP.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
app = AspectAwarePreprocessor(224, 224)


ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--cascade", required=True,help="path to where the face cascade resides")
# ap.add_argument("-m", "--model", required=True,help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",help="path to the (optional) video file")
args = vars(ap.parse_args())


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model('without_aug_mobilenet_mask_model.hdf5')
if not args.get("video", False):
    camera = cv2.VideoCapture(1)

    # otherwise, load the video
else:
    camera = cv2.VideoCapture()




# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # print("here1")
    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # print("here2")

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we can draw on it later in the program
    # frame = imutils.resize(frame, width=300)
    frameClone = frame.copy()

    rects = detector.detectMultiScale(frame, scaleFactor=1.1,minNeighbors=5, minSize=(200, 200),flags=cv2.CASCADE_SCALE_IMAGE)
    # print("here3")
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = frame[fY:fY + fH, fX:fX + fW]
        # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # roi = cv2.resize(roi, (224, 224))
        roi = img_to_array(roi)

        roi = app.preprocess(roi)
        roi = roi.astype("float")/255.0
        roi = np.expand_dims(roi, axis=0)
        print(roi.shape)








        # roi = cv2.resize(roi, (28, 28)
        # roi = roi.astype("float") / 255.0
        # roi = img_to_array(roi)
        # roi = np.expand_dims(roi, axis=0)
        # determine the probabilities of both "smiling" and "not
        # smiling", then set the label accordingly
        (mask, no_mask) = model.predict(roi)[0]
        label = "Mask {}".format(mask*100) if mask > no_mask else "No Mask {}".format(no_mask*100)
        color = (0, 255, 0) if mask > no_mask else (0, 0, 255)



        cv2.putText(frameClone, label, (fX, fY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),color, 2)

        cv2.imshow("Face", frameClone)

# if the ’q’ key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
