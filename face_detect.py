import dlib
from skimage import io
from skimage.draw import polygon_perimeter

detector = dlib.get_frontal_face_detector()
sample_image = io.imread('/Users/zhangwenjin/Documents/test.png')
faces = detector(sample_image, 2)

for d in faces:
    rr, cc = polygon_perimeter([d.top(), d.top(), d.bottom(), d.bottom()], [d.right(), d.left(), d.left(), d.right()])
    sample_image[rr, cc] = (0, 255, 0)

io.imsave('/Users/zhangwenjin/Documents/test2.png', sample_image)