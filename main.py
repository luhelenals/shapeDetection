import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detec_circles(file, k, minR, maxR):
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
    gray = cv.medianBlur(gray, k)
    ## [reduce_noise]

    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows/16,
                               param1=100, param2=30,
                               minRadius=minR, maxRadius=maxR)
    ## [houghcircles]

    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (255, 0, 0), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    else:
        print('no circles :(')
    ## [draw]

    return src

def display_img(src):
    # Exibir com Matplotlib
    plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def detec_ellipses(file):
    # Caminho da imagem carregada
    bgr_img = cv.imread(file)  # Lê a imagem

    # Converta a imagem para escala de cinza
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    # Suavizar imagem e aplicar binarização
    blur = cv.GaussianBlur(gray_img,(9,9),0)
    th3 = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
    
    # Aplicar um filtro de Canny para detecção de bordas
    edges = cv.Canny(th3, threshold1=90, threshold2=175, apertureSize=3)

    # Criar uma cópia da imagem original para desenhar os resultados
    result_img = np.copy(bgr_img)

    # Detectar elipses usando contornos
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) >= 5:  # Pelo menos 5 pontos são necessários para ajustar uma elipse
            ellipse = cv.fitEllipse(contour)

            # Verificar se largura e altura da elipse são válidas
            (x, y), (width, height), angle = ellipse
            if width > 15 and height > 15 and width < 80 and height < 80:  # Determinar tamanhos mínimos e máximos
                cv.ellipse(result_img, ellipse, (255, 0, 0), 3)

    return result_img

def main():
    op = int(input('OPTIONS:\n1 - Circle detection\n'))
    if op == 1:
        path = input('Caminho da imagem: ')
        k = input('Tamanho do kernel para suavização: ')
        minR = input('Raio mínimo dos círculos: ')
        maxR = input('Raio mínimo dos círculos: ')
        display_img(detec_circles(path, k, minR, maxR))
    elif op == 2:
        path = input('Caminho da imagem: ')
        size = input('Tamanho máximo das elipses: ')
        display_img(detec_ellipses(path, size))
    else:
        print('Invalid option')

if __name__ == "__main__":
    main()