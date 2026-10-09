import json
import time
import requests

# TEST_INPUT_FILE = 'test_image.jpg'
TEST_INPUT_FILE = 'test_image_10pc.jpg'

# url = 'http://b3104:8005/preview/'
# url = "http://207.189.112.61:31605/preview"
url = 'http://localhost:8000/preview/'

# url = 'https://funnel.tailce84f.ts.net/preview/'
style_to_filename = {
    "anime_2_5d": "1_2.5d_anime_model.json",
    "anime_2d_flat": "2_2d_flat_anime_model.json",
    "cartoon_3d": "3_3d_cartoon_style.json",
    "comic_book": "4_comic_book_model.json",
    "anime_ghibli": "5_ghibli_anime_model.json",
    "ink_punk": "6_ink_punk.json",
    "ink_splash": "7_ink_splash.json",
    "ink_bw_style": "8_ink_w_and_b_style.json",
    "jojo_style": "9_jojo_style.json",
    "paper_origami": "10_paper_origami.json",
    "pixel_art": "11_pixel_art.json",
    "pop_art": "12_pop_art.json",
    "realistic_1": "13_realistic_1.json",
    "realistic_2": "14_realistic_2.json",
    "anime_retro_neon": "15_retro_neon_anime_90.json",
    "anime_standard": "16_standard_anime_model.json",
    "hr_giger": "17_hr_giger.json",
    "simpsons": "18_simpsons.json",
    "carnage": "19_carnage.json",
    "pastel_cute_anime": "20_pastel_cute_anime.json",
    "bloom_lighting": "21_bloom_lighting.json",
    "25d_horror": "22_25D_Horror.json",
    "creepy": "23_creepy.json",
    "creepy_vhs": "24_creepy_vhs.json",
    "trail_cam_footage": "25_trail_cam_footage.json",
    "old_black_white_movie": "26_old_black_white_movie.json",
    "horror_noir_black_white": "27_horror_noir_black_white.json",
    "techno_noir_black_white": "28_techno_noir_black_white.json",
    "black_white_20s": "29_black_white_20s.json",
    "cyberpunk_anime": "30_cyberpunk_anime.json",
    "dragonball": "31_dragonball.json",
    "realistic_matrix": "32_realistic_matrix.json",
    "realistic_cyberpunk": "33_realistic_cyberpunk.json",
    "dreamer": "34_dreamer.json",
}

all_styles = list(style_to_filename.keys())

import random
random.shuffle(all_styles)

styles = list(all_styles)[:]
# random.shuffle(styles)

test_style_list = styles + styles
# for style, filename in list(style_to_filename.items())[:2]:
for style in all_styles:
    for i in range(1):
        # print(f"Testing style: {style}")
        payload = {"style": style, "positive_prompt": "test pos in", "negative_prompt": "test neg in"}
        with open(TEST_INPUT_FILE , 'rb') as input_file:
            # files = {'input_file': ('input_file', input_file, 'image/jpeg')}
            files = {'input_file':  input_file}
            # write output file suffixed with timestamp
            timestamp = time.strftime('%Y%m%d%H%M%S')
            output_file_name = f'outputs/output_image_{style}_{timestamp}.jpeg'
            execution_start_time = time.perf_counter()
            with requests.post(url, data={'request': json.dumps(payload)}, files=files) as response:
                if response.status_code == 200:
                    with open(output_file_name, 'wb') as f:
                        f.write(response.content)
                else:
                    # Handle request error
                    print(f"Request failed: {response.status_code}")
                    print(response.text)
            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            print('{} generated {} in {:.2f} seconds'.format(style, output_file_name, execution_time))
