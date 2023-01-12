from utils import *
import yaml
import os
import ast

def main(image_path, filtering_frequency_noise, denoise, thresh_edge, thresh_lines, thresh_board, thresh_lines_2):
    chessboard_size = None
    
    if filtering_frequency_noise:

        chessboard_image = read_image(image_path)

        filtering_image = filter_frequency_noise(chessboard_image, 1)

        normalized_image = normalize(filtering_image)

        brightness_image = lighten_and_denoise_image(normalized_image)

        gray_image = convert_BGR2GRAY_and_denoise(brightness_image)

        edge_image = detect_edge(gray_image, thresh_edge)

        output = find_contour(chessboard_image, edge_image)

        corner_points = find_chessboard_region(chessboard_image, output, thresh_lines)

        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)

        chessboard_region = extract_chessboard_region(normalized_image, corner_points)

        chessboard_size = get_chessboard_size(normalized_image, chessboard_region, thresh_board, thresh_lines_2)
    
    else:
        
        chessboard_image = read_image(image_path)
        
        if denoise:
            chessboard_image = denoise_image(chessboard_image)

        brightness_image = lighten_and_denoise_image(chessboard_image)

        gray_image = convert_BGR2GRAY_and_denoise(chessboard_image)

        gray_brightness_image = convert_BGR2GRAY_and_denoise(brightness_image)

        edge_image = detect_edge(gray_image, thresh_edge)

        output = find_contour(chessboard_image, edge_image)

        corner_points = find_chessboard_region(chessboard_image, output, thresh_lines)

        chessboard_region = extract_chessboard_region(gray_brightness_image, corner_points)

        chessboard_size = get_chessboard_size(chessboard_image, chessboard_region, thresh_board, thresh_lines_2)
                                              
    return chessboard_size

if __name__ == "__main__":
    data_dir = './data/'
    
    image = 'Chessboard_00451_2.png'

    with open("config.yml", "r", encoding="utf8") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    image_path = data_dir + image
            
    cfg = config[image]
            
    denoise = cfg['denoise']
            
    filtering_frequency_noise = cfg['filtering_frequency_noise']
    
    thresh_edge = ast.literal_eval(cfg['thresh_edge'])

    thresh_lines = cfg['thresh_lines']

    thresh_board = ast.literal_eval(cfg['thresh_board'])

    thresh_lines_2 = cfg['thresh_lines_2']
            
    chessboard_size = main(image_path, filtering_frequency_noise, denoise, thresh_edge, thresh_lines, thresh_board, thresh_lines_2)
            
    print(f'Size of the chessboard is: {chessboard_size[0]} x {chessboard_size[1]}')
    
#     for image in os.listdir("./data"):
#         if image.endswith('.png'):
#             image_path = data_dir + image
            
#             cfg = config[image]
            
#             denoise = cfg['denoise']
            
#             filtering_frequency_noise = cfg['filtering_frequency_noise']
    
#             thresh_edge = ast.literal_eval(cfg['thresh_edge'])

#             thresh_lines = cfg['thresh_lines']

#             thresh_board = ast.literal_eval(cfg['thresh_board'])

#             thresh_lines_2 = cfg['thresh_lines_2']
            
#             chessboard_size = main(image_path, filtering_frequency_noise, denoise, thresh_edge, thresh_lines, thresh_board, thresh_lines_2)
            
#             print(chessboard_size)
            
            
            
