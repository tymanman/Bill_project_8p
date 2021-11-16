# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
 
def rad(x):
    return x * np.pi / 180
 
 
def rotate_3(img,keypoints=None,angle_vari=15):
    h, w = img.shape[0:2]
    fov = 42
    anglex = np.random.uniform(-angle_vari, angle_vari)
    # anglex = 0
    angley = np.random.uniform(-angle_vari, angle_vari)
    # anglez = np.random.uniform(-angle_vari+10, angle_vari-10)
    anglez = np.random.uniform(-150, 150)
    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)
 
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)
 
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)
 
    r = rx.dot(ry).dot(rz)
 
    # 四对点的生成
    pcenter = np.array([w / 2, h / 2, 0, 0], np.float32)
 
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
 
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
 
    list_dst = [dst1, dst2, dst3, dst4]
 
    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)
    org_ = np.array([[0, 0, 1],
                    [h, 0, 1],
                    [0, w, 1],
                    [h, w, 1]], np.float32)
    dst = np.zeros((4, 2), np.float32)
 
    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
 
    warpR = cv2.getPerspectiveTransform(org, dst)
    new_shape = (int(dst[:, 0].max() - dst[:, 0].min()), int(dst[:, 1].max() - dst[:, 1].min()))
    offset = (int(dst[:, 0].min()), int(dst[:, 1].min()))
    warpR[0, :] -=  warpR[2, :]*offset[0]
    warpR[1, :] -=  warpR[2, :]*offset[1]
    return_keyps = []
    if keypoints:
        for keyp in keypoints:
            keypoint = np.array(keyp["keypoint"])
            keypoint_3dim =  np.concatenate((keypoint, np.ones((4, 1))), 1).transpose(1,0)
            project_keyp = np.matmul(warpR, keypoint_3dim)
            project_keyp[0, :] /= project_keyp[2, :]
            project_keyp[1, :] /= project_keyp[2, :]
            return_keyps.append(project_keyp[:2, :].transpose(1, 0).tolist())
        
    return_img = cv2.warpPerspective(img, warpR, new_shape,borderMode=cv2.BORDER_REPLICATE)
 
    return (return_img, return_keyps) if keypoints else return_img
 
 
 
def rotate(image, angle_vari=30):
     angle = np.random.uniform(-angle_vari, angle_vari)
     rows, cols = image.shape[:2]
     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
     dst = cv2.warpAffine(image, M, (cols, rows))
     return dst
 
if __name__ == '__main__':
    root_dir = "/Users/lili/Workdir/code/smiles_project/card_det_project/data/train_val_data/val/imgs/"
    new_dir = "new_val_imgs"
    imgs = os.listdir(root_dir)
    num_imgs = len(imgs)
    angle_vari = 20
    cnt = 0
    while True:
        name = imgs[cnt%num_imgs]
        img = cv2.imread(os.path.join(root_dir, name))
        result=rotate_3(img, angle_vari=angle_vari)
        print(result.shape)
        cv2.imshow("result", result)
        c = cv2.waitKey(0)
        if c == ord("q"):
            break
        elif c == ord("s"):
            cv2.imwrite(os.path.join(new_dir, name), img)
        cnt+=1

