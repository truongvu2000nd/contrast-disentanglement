import cv2


if __name__ == '__main__':
    img = cv2.imread("../../datasets/DFW2018/Training_data/0604/0604.jpg")
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()