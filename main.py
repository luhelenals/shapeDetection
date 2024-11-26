import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detec_circles(file):
    # Loads an image
    src = cv.imread(cv.samples.findFile(file), cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + file + '] \n')
        return -1
    ## [load]

    ## [convert_to_gray]
    # Convert it to gray
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = cv.medianBlur(gray, 17)
    ## [reduce_noise]

    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows/16,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=200)
    ## [houghcircles]

    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    else:
        print('no circles :(')
    ## [draw]

    ## [display]
    # Exibir com Matplotlib
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.title("Imagem")
    plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    ## [display]

    return 0

def main():
    op = int(input('OPTIONS:\n1 - Circle detection\n'))
    if op == 1:
        circles_path = 'c:/Users/luiza/Hough/laranja.jpg'
        detec_circles(circles_path)
    else:
        print('Invalid option')

if __name__ == "__main__":
    main()