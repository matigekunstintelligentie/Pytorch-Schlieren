import cv2

video = "MAH04536.MP4"

cap = cv2.VideoCapture(video)

frame_list = []
i = 0
while(True):
    ret, frame = cap.read()

    if not ret:
        break

    frame_list.append(frame)

    # print(i)
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break
    # i += 1

cap.release()
cv2.destroyAllWindows()

cv2.imwrite("original.png", frame_list[0])
cv2.imwrite("fire.png", frame_list[144])
