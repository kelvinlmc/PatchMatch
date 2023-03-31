import numpy as np
from PIL import Image

def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    patch_a = A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]  #[[[元素]]]
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :]
    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp))) #nn.nan是arry独特的占位符，没有值就是nan，np.isnan就是判断是否是nan，会遍历所有元素
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist

def initialization(A, B, p_size):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    random_B_r = np.random.randint(p, B_h-p, [A_h, A_w])  #随机生成最小值p最大值B_h-p形状为[A_h, A_w]的数组
    random_B_c = np.random.randint(p, B_w-p, [A_h, A_w])
    A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan  #长宽填充增加2就是长宽都加2
    A_padding[p:A_h+p, p:A_w+p, :] = A  #把不是填充的部分用A的值来取代
    f = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])
    for i in range(A_h):
        for j in range(A_w):
            a = np.array([i, j])   #A当中的元素当前是随机整数
            b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)  #给b复制
            f[i, j] = b
            dist[i, j] = cal_distance(a, b, A_padding, B, p_size)
    return f, dist, A_padding

img = np.array(Image.open("./dog.jpg"))
ref = np.array(Image.open("./cup_a.jpg"))
initialization(img,ref,3)
A_H = img.shape[0]
A_W = img.shape[1]
nnf = np.zeros([A_H, A_W, 2])
#print(nnf)
#a=np.zeros([2, 3, 4])
#print(a)
'''p_size=3
A_h = np.size(img, 0)
A_w = np.size(img, 1)
p = p_size // 2
k=np.ones([A_h+p*2, A_w+p*2, 3])
A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
c = np.random.randint(1, 10, [8,7 ])'''
#print(A_h,A_w)
#print(k)
#print(A_padding.shape)
#print(c)
'''k=np.array([[[1,2,3],
             [2,3,5],
             [2,8,3]]])
print(k.shape)
print(k[0:1,0:2,1:2])'''