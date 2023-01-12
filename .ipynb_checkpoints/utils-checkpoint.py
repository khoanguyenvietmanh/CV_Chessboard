import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils import perspective
from imutils import contours
import math
from math import sqrt
from collections import defaultdict


def read_image(path):
    image = cv2.imread(path)
    return image

def denoise_image(image):
    noiseless_image_colored = cv2.fastNlMeansDenoisingColored(image , None, 20, 20, 8, 21)
    return noiseless_image_colored

def lighten_and_denoise_image(image):
    B, G, R = cv2.split(image)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    out = cv2.merge((B,G,R))

    noiseless_image_colored = cv2.fastNlMeansDenoisingColored(out , None, 20, 20, 8, 21)
    
    return noiseless_image_colored


def convert_BGR2GRAY_and_denoise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    return blur

def detect_edge(blur, thresh_edge):
    # perform edge detection
    edge = cv2.Canny(blur, thresh_edge[0], thresh_edge[1])
    edge = cv2.dilate(edge, None, iterations=1)
    edge = cv2.erode(edge, None, iterations=1)
    
    return edge

def find_contour(image, edge):
    output = image.copy()
    
    cnts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(output, [c], -1, (255, 255, 255), 2)
    
    return output

def compute_distance(p, p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    num = abs(y_diff * p[0] - x_diff * p[1] + p2[0]*p1[1] - p2[1]*p1[0])
    den = math.sqrt(y_diff**2 + x_diff**2)
    return num * 1.0/ den

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return math.floor(x), math.floor(y)


def find_chessboard_region(image, contour_image, thresh_lines):
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    
    contour_image[np.where(contour_image != 255)] = 0
    lines = cv2.HoughLinesP(contour_image.copy(), rho=1, theta=np.pi/180, threshold=80, minLineLength = thresh_lines, maxLineGap = 100)

    output = image.copy()

    horizontal_midpoint = (0, output.shape[1] // 2 - 100)
    vertical_midpoint = (output.shape[0] // 2 + 150, 0)

    hor_line_list = []
    ver_line_list = []
    horizontal_list = []
    vertical_list = []
    list = []

    limit = 30

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(k, (x1, y1), (x2, y2), (0, 0, 255), 2)
        spl = math.degrees(math.atan2((y1 - y2), (x1 - x2))) % 360

        if (not limit < spl <= 360 - limit) or (180 - limit <= spl < 180 + limit):
            hor_line_list.append([(x1, y1), (x2, y2)])
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif (90 - limit < spl < 90 + limit) or (270 - limit < spl < 270 + limit):
            ver_line_list.append([(x1, y1), (x2, y2)])
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


    for hor_line in hor_line_list:
        x1, y1 = hor_line[0]
        x2, y2 = hor_line[1]
        horizontal_list.append(compute_distance(vertical_midpoint, (x1, y1), (x2, y2)))

    for ver_line in ver_line_list:
        x1, y1 = ver_line[0] 
        x2, y2 = ver_line[1]
        vertical_list.append(compute_distance(horizontal_midpoint, (x1, y1), (x2, y2)))


    horizontal_list = np.array(horizontal_list)
    vertical_list = np.array(vertical_list)
    list = np.array(list)

    min_ho, max_ho = np.argmin(horizontal_list), np.argmax(horizontal_list)
    min_ver, max_ver = np.argmin(vertical_list), np.argmax(vertical_list)

    a_1 = line_intersection(hor_line_list[min_ho], ver_line_list[min_ver])
    b_1 = line_intersection(hor_line_list[min_ho], ver_line_list[max_ver])
    c_1 = line_intersection(hor_line_list[max_ho], ver_line_list[min_ver])
    d_1 = line_intersection(hor_line_list[max_ho], ver_line_list[max_ver])
        
    return a_1, b_1, c_1, d_1


def extract_chessboard_region(gray_image, points):
    a_1, b_1, c_1, d_1 = points
    
    mask = np.zeros((gray_image.shape), np.uint8)
    
    cv2.line(mask, a_1, b_1, (255, 255, 255), 3)
    cv2.line(mask, c_1, d_1, (255, 255, 255), 3)
    cv2.line(mask, b_1, d_1, (255, 255, 255), 3)
    cv2.line(mask, a_1, c_1, (255, 255, 255), 3)
    
    mask = cv2.rectangle(mask, (0, 0), (mask.shape[1], mask.shape[0]), 255, 1)
    
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask, [cnts[0]], 0, 255, -1)
    
    output = np.zeros_like(gray_image)

    output[mask == 255] = gray_image[mask == 255]
    
    return output


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec
    
    vertical_lines = []
    horizontal_lines = []
    # segment lines based on their kmeans label
    for i, line in enumerate(lines):
        if labels[i] == 0:
            vertical_lines.append(line)
        else:
            horizontal_lines.append(line)
    #     segmented[labels[i]].append(line)
    # segmented = list(segmented.values())
    return vertical_lines, horizontal_lines


def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [[x0, y0]]


def count_line(hori_points, ver_points):
    count_hori = 1
    i = 1

    p = hori_points[0]
    while i < len(hori_points):
        check = hori_points[i]
    
        if check[0] - p[0] > 15:
            count_hori += 1
            p = hori_points[i]
    
        i += 1
        
    count_ver = 1
    i = 1

    p = ver_points[0]
    while i < len(ver_points):
    
        check = ver_points[i]

        if check[1] - p[1] > 15:
            count_ver += 1
            p = ver_points[i]

        i += 1
        
    return count_ver - 1, count_hori - 1


def get_chessboard_size(chessboard_image, chessboard_region, thresh_board, thresh_lines_2):
    image = cv2.cvtColor(chessboard_region, cv2.COLOR_GRAY2BGR)
    
    img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (13, 13), 0)
    edged = cv2.Canny(gray, thresh_board[0], thresh_board[1])
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # lines_2 = cv2.HoughLinesP(final_gray.copy(), rho=1, theta=np.pi/180, threshold = 100, minLineLength = 200, maxLineGap = 100)
    lines_2 = cv2.HoughLines(edged.copy(), rho=1, theta=np.pi/180, threshold=thresh_lines_2)

    for line in lines_2:
        rho, theta = line[0]
        # x1, y1, x2, y2 = line[0]
        # x = np.asarray([[x1, x2]], dtype = "float64")
        # y = np.asarray([[y1, y2]], dtype = "float64")
        # rho, theta = cv2.cartToPolar(x, y, angleInDegrees=False)
        # print(rho, theta)
        # rho, theta = rho[0][0], theta[0][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
    vertical_lines, horizontal_lines = segment_by_angle_kmeans(lines_2)
    img = chessboard_image.copy()

    horizontal_midpoint = (0, img.shape[1] // 2 - 100)
    horizontal_midpoint_end = (img.shape[0] * 2, img.shape[1] // 2 - 100)
    vertical_midpoint = (img.shape[0] // 2 + 150, 0)
    vertical_midpoint_end = (img.shape[0] // 2 + 150, img.shape[1] * 2)

    x = np.asarray([[horizontal_midpoint[0], vertical_midpoint[0]]], dtype = "float64")
    y = np.asarray([[horizontal_midpoint[1], vertical_midpoint[1]]], dtype = "float64")

    mag, angle = cv2.cartToPolar(x, y, angleInDegrees=False)

    hori_line = (mag[0][0], angle[0][0])
    ver_line = (mag[0][1], angle[0][1])

    hori_points = []
    ver_points = []

    for line in vertical_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1, y1),(x2, y2), (0,0,255), 2)
        intersec_point_ver = intersection(ver_line, (rho, theta))
        ver_points.append(intersec_point_ver[0])


    for line in horizontal_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1, y1),(x2, y2), (0,0,255), 2)
        intersec_point_hori = intersection(hori_line, (rho, theta))
        hori_points.append(intersec_point_hori[0])


    hori_points = sorted(hori_points, key=lambda x: x[0])
    ver_points = sorted(ver_points, key=lambda x: x[1])
    
    return count_line(hori_points, ver_points)


def butterworthLP(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def filter_frequency_noise(image, val):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fourier_transform = np.fft.fft2(img)
    center_shift = np.fft.fftshift(fourier_transform)

    fourier_noisy = 20 * np.log(np.abs(center_shift))

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    if val == 1:
        # horizontal mask
        center_shift[crow - 4:crow + 4, 0:ccol] = 1
        center_shift[crow - 4:crow + 4, ccol:] = 1

    elif val == 2:
        # vertical mask
        center_shift[:crow, ccol - 4:ccol + 4] = 1
        center_shift[crow:, ccol - 4:ccol + 4] = 1

    elif val == 3:
        # diagonal-1 mask
        for x in range(0, rows):
            for y in range(0, cols):
                if (x == y):
                    for i in range(0, 10):
                        center_shift[x - i, y] = 1
    elif val == 4:
        # diagonal-2 mask
        for x in range(0, rows):
            for y in range(0, cols):
                if (x + y == cols):
                    for i in range(0, 10):
                        center_shift[x - i - 160, y] = 1


    filtered = center_shift * butterworthLP(80, img.shape, 10)

    f_shift = np.fft.ifftshift(center_shift)
    denoised_image = np.fft.ifft2(f_shift)
    denoised_image = np.real(denoised_image)

    f_ishift_blpf = np.fft.ifftshift(filtered)
    denoised_image_blpf = np.fft.ifft2(f_ishift_blpf)
    denoised_image_blpf = np.real(denoised_image_blpf)

    fourier_noisy_noise_removed = 20 * np.log(np.abs(center_shift))
    
    return denoised_image_blpf


def normalize(image):
    normalized_image = (image-np.min(image)) / (np.max(image) - np.min(image))
    normalized_image = np.uint8(255 * normalized_image)
    
    bgr_normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
    
    return bgr_normalized_image 
    


    