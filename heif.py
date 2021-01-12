import os
#import pyheif
import PIL
from PIL import Image
import numpy as np

file_list = os.listdir("./pill_dataset_init")
answer = {
    '네렉손서방정': 0,
    '레피콜에이': 1,
    '로도질정': 2,
    '록소드펜정': 3,
    '아나프록스정': 4,
    '오티렌F정': 5,
    '타이레놀': 6,
    '펠루비서방정': 7,
    '화이투벤': 8
}
answer_list = []
i = 0


#image change
for name in file_list:
    if name != '.DS_Store':
        img_name_list = os.listdir("pill_dataset_init/" + name)
        for img_name in img_name_list:
            if img_name != '.DS_Store':
                image = Image.open("pill_dataset_init/" + name + '/' + img_name)
                image = image.resize((84, 84))
                image.save("pill_dataset/" + "%03d"%i + '.jpg')
                print(i)
                answer_list.append(answer[name])
                answer_np = np.asarray(answer_list)
                np.save("pill_dataset_answer", answer_np)
                i += 1
            pass
    continue