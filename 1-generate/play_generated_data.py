import cv2

for match in [1,2,3,4,5,6,7,8,9,10]:

    img_array = []

    for frame in range(4500):
        img = cv2.imread(f'data/match_{match}/img/{frame}.png')
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(f'{match}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()