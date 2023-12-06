###### ########
# 새로운 사람 할때마다 cropped 파일 지워야함
# update_palette 함수 >> mode= recommend 일때만 user palette 업데이트 하도록 수정해야함
# 영역 값 조절해서 색 덜빠지게 하기
########
from gensim.models import FastText
from skimage.transform import resize
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance as ssd

import random
import sys
import utils
import cv2
import numpy as np
import pandas as pd
import pygame
import pygame.camera
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import threading
import pyautogui
from PIL import Image, ImageEnhance

IMAGE_SIZE = 256
FONT_NAME = './PokemonGb-RAeo.ttf'
BGCOLOR = (0, 0, 0)

KEY_4 = 1073741916
KEY_6 = 1073741918
KEY_5 = 1073741917
KEY_2 = 1073741914
KEY_8 = 1073741920
KEY_0 = 1073741922

SEGCOLOR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
            (0, 0, 192), (0, 192, 0), (192, 0, 0), (0, 192, 128), (192, 192, 0), (192, 0, 192),
            (0, 0, 128), (0, 128, 0), (128, 0, 0), (0, 128, 128), (128, 128, 0), (128, 0, 128),
            (0, 0, 64), (0, 64, 0), (64, 0, 0), (0, 64, 64), (64, 64, 0), (64, 0, 64),
            (0, 0, 32), (0, 32, 0), (32, 0, 0), (0, 32, 32), (32, 32, 0), (32, 0, 32)]

LOG_DF = pd.DataFrame(data=None,
                      columns=['Time', 'Mode', 'User_man', 'Recommended_man', 'Recommended_layer_id', 'Keyboard_input'])
LOG_INDEX = 0

WORKING = 0


## 만다라 처리 클래스
class Mandala:
    def __init__(self, image):
        self.org_mandala = image
        self.aft_mandala = None
        self.height, self.width = self.org_mandala.shape[0], self.org_mandala.shape[1]
        self.cids, self.cluster = None, None
        self.edge_image = None
        self.mask = None
        self.masked = None
        self.coord = None
        self.labels, self.prop_labels, self.area_crop = None, None, None
        self.sim_mat = None
        self.analyze_mandala()

    def mean_iou(self, y_true, y_pred):  # 모델에 사용되는 loss
        intersection = tf.reduce_sum(tf.where((y_pred > 0.5) & (y_true == 1), 1, 0))
        union = tf.reduce_sum(tf.where((y_pred > 0.5) | (y_true == 1), 1, 0))
        return intersection / union

    def extend_canvas(self, ori, symmetry=False):  # 만다라에 여백 만들기
        h, w = ori.shape[:2]
        img = np.ones((h + 50, w + 50, 3), dtype='uint8') * 255
        img[25:h + 25, 25:w + 25, :] = ori

        h, w = img.shape[:2]

        if symmetry == True:
            img[h // 2:, w // 2:, :] = img[:h // 2, :w // 2, :][::-1, ::-1, :]
            img[h // 2:, :w // 2, :] = img[:h // 2, :w // 2, :][::-1, :, :]
            img[:h // 2, w // 2:, :] = img[:h // 2, :w // 2, :][:, ::-1, :]
        return img

    def get_partitions(self, img, masked):
        global star
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(masked, connectivity=4)
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgh, imgw = img.shape[0], img.shape[0]
        area_crop, prop_labels, coord = [], [], []
        unique, counts = np.unique(labels, return_counts=True)

        for i in unique:
            tmp = np.zeros_like(gimg)
            tmp[labels == i] = 255

            x, y, w, h = cv2.boundingRect(tmp)
            if np.sum(labels == i) < 30 or w < 2 or h < 2 or y < 2 or x < 2 or y + h > imgh - 2 or x + w > imgw - 2:
                continue
            area_crop.append(tmp[y:y + h, x:x + w])
            prop_labels.append(i)
            coord.append(np.sqrt(((y + h / 2) / imgh - 0.5) ** 2 + ((x + w / 2) / imgw - 0.5) ** 2))

        self.labels, self.prop_labels, self.area_crop, self.coord = labels, prop_labels, area_crop, coord

    def repaint(self, used_colors, current, segcolor):

        out = np.ones_like(self.edge_image) * 255
        for i, c in enumerate(self.cluster):
            out[self.labels == self.prop_labels[i]] = segcolor[int(c % len(segcolor))]
        out[np.where((self.edge_image == (0, 0, 0)).all(axis=-1))] = (0, 0, 0)

        for c in used_colors:
            target_area_c = np.median(out[(current == c).all(axis=-1)].reshape((-1, 3)), axis=0)
            if ~((target_area_c == c).all(axis=-1)):
                target_area_idx = (current == c).all(axis=-1)
                out[target_area_idx] = c
        return out

    def color_partitions(self, current=None, colorlist=None, num_classes=14, mode='recommendation'):
        if mode == 'recommendation':
            if current is None:
                current = self.edge_image.copy()
            num_areas = len(self.area_crop)
            sim_mat = np.zeros((num_areas, num_areas))
            label_mean = np.array([current[self.labels == i].mean(0) for i in self.prop_labels])

            for i in range(0, num_areas):
                for j in range(0, num_areas):
                    if i >= j:
                        continue
                    elif (label_mean[i] == (255, 255, 255)).all(axis=-1) or \
                            (label_mean[j] == (255, 255, 255)).all(axis=-1):
                        im1 = np.array(self.area_crop[i])
                        im2 = np.array(self.area_crop[j])
                        sim_mat[i][j] = cv2.matchShapes(im1, im2, cv2.CONTOURS_MATCH_I2, 0) + 0.5 * abs(
                            self.coord[i] - self.coord[j])
                    elif (label_mean[i] == label_mean[j]).all(axis=-1):
                        sim_mat[i][j] = 0.01
                    else:
                        sim_mat[i][j] = 1000

                    sim_mat[j][i] = sim_mat[i][j]
            Zd = linkage(ssd.squareform(sim_mat), method="complete")

            self.cluster = fcluster(Zd, num_classes, criterion="maxclust") - 1

            self.cids, _ = np.unique(self.cluster, return_counts=True, axis=0)
            self.sim_mat = sim_mat

        used_colors = np.unique(current.reshape((-1, 3)), axis=0)
        used_colors = used_colors[~((used_colors == (255, 255, 255)).all(axis=-1))]
        colored_mandalas = []
        if colorlist is None:
            segcolor = SEGCOLOR
            out = self.repaint(used_colors, current, segcolor)
            colored_mandalas.append(([255, 255, 255] * 6, out))
        else:
            for key in colorlist.keys():
                ip_color, rc_color = colorlist[key][0], colorlist[key][1] + [[-1, -1, -1]]
                if len(ip_color) >= 5:
                    ip_color = ip_color[:5]
                ip_color = random.sample(ip_color, random.randint(0, min(len(ip_color) - 1, 2)))
                idx, cnt = 0, 6 - len(ip_color)
                while idx + cnt < len(rc_color):
                    shuffled = rc_color[idx:idx + cnt]
                    random.shuffle(shuffled)
                    segcolor = ip_color + shuffled
                    out = self.repaint(used_colors, current, segcolor)
                    colored_mandalas.append((ip_color + shuffled, out))
                    idx += 1
        random.shuffle(colored_mandalas)
        return colored_mandalas

    def analyze_mandala(self):
        extended = self.extend_canvas(self.org_mandala, symmetry=True)
        # get_edge
        gimg = cv2.cvtColor(extended, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(255 - gimg, 50, 255, cv2.THRESH_BINARY)
        edge_image = np.stack((255 - th,) * 3, axis=-1)
        self.edge_image = edge_image

        # get_mask
        gimg = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
        _, th2 = cv2.threshold(255 - gimg, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.ones_like(th2) * 255
        cv2.drawContours(mask, contours, -1, 0, -1)
        self.mask = mask

        masked = cv2.bitwise_and(255 - th2, 255 - th2, mask=255 - mask)

        masked2 = np.stack((masked,) * 3, axis=-1).reshape(-1, 3)
        alist, a = np.unique(masked2, return_counts=True, axis=0)

        self.allblack = a[0]  # 배경 테두리 검정색값
        self.masked = masked
        self.get_partitions(edge_image, masked)
        self.color_partitions()


###############################################################


##만다라 색상추천 클래스
class Coloring:
    def __init__(self, ip_colors=None):
        if ip_colors is None:
            ip_colors = [[255, 255, 255]] * 6
        self.model = FastText.load('model/col2vec_fasttext.model')
        self.ip_colors = ip_colors
        self.colorlist = {}

    def rgb_to_hex(self, color):
        r, g, b = color
        return (hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)).upper()

    def hex_to_rgb(self, value):
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def recommend_palette(self, idx, state, color_info):
        ratio, used_colors = zip(*color_info)
        w_lis = None
        if state == 'initial':
            w_lis = [self.rgb_to_hex(used_colors[1])]
        elif state == 'colorized':
            w_lis = [self.rgb_to_hex(c) for r, c in zip(ratio[1:], used_colors[1:]) if r / sum(ratio[1:]) >= 0.1]

        recommend_w_colors = self.model.wv.most_similar(w_lis)
        recommend_w_pal = [list(self.hex_to_rgb(rc[0])) for rc in recommend_w_colors]

        self.colorlist[idx] = {'w_palette': ([list(self.hex_to_rgb(c)) for c in w_lis], recommend_w_pal)}


##만다라 변경사항 업데이트 클래스
class UpdateMandala:
    def __init__(self, image):
        self.mandala = Mandala(image)
        self.colors = Coloring()
        self.color_info = [[1, [125, 125, 125]]]  # zip(ratio, color RGB)
        self.user_mandala = None
        self.colored_mandalas = None

    def ratio_used_colors(self, used_colors_cnts):  # current_quant, mask
        used_colors_cnts[0] = used_colors_cnts[0] - self.mandala.allblack
        if used_colors_cnts[0] <= 0:
            used_colors_cnts[0] = 0
        per = used_colors_cnts / np.array(used_colors_cnts).sum()
        per = per.round(3)
        return per

    def quantization(self, current):
        extended = self.mandala.extend_canvas(current, symmetry=False)
        coloring = self.mandala.edge_image.copy()

        used_colors = []
        for l in self.mandala.prop_labels:
            area = extended[self.mandala.labels == l]
            uqe = np.unique(extended[self.mandala.labels == l] // 25, axis=0, return_counts=True)
            if (np.sum(self.mandala.labels == l) > 100 or np.mean(np.std(area, axis=-1)) > 50) and np.sum(
                    uqe[1][(uqe[0] == (10, 10, 10)).all(axis=-1)]) / np.sum(uqe[1]) < 0.3:
                coloring[self.mandala.labels == l] = area
                c = np.mean(area[~(area == (255, 255, 255)).all(axis=-1)], axis=0).astype(np.uint8)
                coloring[self.mandala.labels == l] = c
                used_colors.append(c)
        if used_colors != []:
            used_colors_hsv = cv2.cvtColor(np.expand_dims(used_colors, axis=0), cv2.COLOR_BGR2HSV)[0]
            coloring_hsv = cv2.cvtColor(coloring, cv2.COLOR_BGR2HSV)

            clt = KMeans(n_clusters=1)
            clt.fit(used_colors)
            baseline = clt.inertia_

            cluster, center = [], []
            for i in range(2, min(len(used_colors), 12)):
                clt = KMeans(n_clusters=i)
                clt.fit(used_colors_hsv)
                cluster = clt.predict(used_colors_hsv)
                center = clt.cluster_centers_.astype(np.int32)

                if (baseline - clt.inertia_) / baseline < 0.1:
                    break
                baseline = clt.inertia_

            for i, c in enumerate(used_colors_hsv):
                coloring_hsv[(coloring_hsv == c).all(axis=-1)] = center[cluster[i]]

            coloring = cv2.cvtColor(coloring_hsv, cv2.COLOR_HSV2BGR)

            used_colors, used_colors_cnts = np.unique(
                (coloring - self.mandala.mask.reshape(coloring.shape[0], coloring.shape[0], 1)).reshape(-1, 3),
                return_counts=True, axis=0)
            color_ratio = self.ratio_used_colors(used_colors_cnts)
        else:
            color_ratio = [1]
        if color_ratio[-1] >= 0.98:
            color_ratio = [0.5, 0.5]
            new_used_colors = [[255, 255, 255]] + [[random.randint(0, 255) for _ in range(3)]]
            self.color_info = list(zip(color_ratio, new_used_colors))
            state = 'initial'  # 색칠하지 않은 스케치가 들어가는 경우
        else:
            self.color_info = sorted(zip(color_ratio, used_colors), key=lambda x: x[0], reverse=True)
            state = 'colorized'  # 유저가 색칠한 스케치가 들어가는 경우
        return state, coloring

    def fill_current_image(self, current):
        out = current.copy()
        for i, l in enumerate(self.mandala.prop_labels):
            c = np.median(current[self.mandala.labels == l], axis=0)

            uqe = np.unique(current[self.mandala.labels == l] // 25, axis=0, return_counts=True)

            # 색칠되어 있으면 ...
            if np.sum(uqe[1][(uqe[0] == (10, 10, 10)).all(axis=-1)]) / np.sum(uqe[1]) < 0.1:
                idx = [k for k in np.argsort(self.mandala.sim_mat[i]) if self.mandala.sim_mat[i][k] < 0.01]
                # 이거랑 비슷한 곳들을 칠함
                for j in idx:
                    c2 = np.median(current[self.mandala.labels == self.mandala.prop_labels[j]], axis=0)
                    if (c2 == (255, 255, 255)).all(axis=-1):
                        out[self.mandala.labels == self.mandala.prop_labels[j]] = c
        return out

    def layered(self, seg):  # 그랩컷으로 단계 만듦
        sl = []
        # new_output에서 조각 정보 얻음
        self.mandala.get_partitions(seg, self.mandala.masked)
        labels, prop_labels, area_crop = self.mandala.labels, self.mandala.prop_labels, self.mandala.area_crop

        for k in range(20, 262, 40):
            s = seg.copy()
            out = np.ones_like(seg) * 255

            mask = np.zeros(seg.shape[:2], dtype='uint8')
            mask2 = np.zeros(seg.shape[:2], dtype='uint8')

            cv2.circle(mask, (int(seg.shape[0] / 2), int(seg.shape[0] / 2)), k + 30, 3, -1)
            cv2.circle(mask, (int(seg.shape[0] / 2), int(seg.shape[0] / 2)), k, 1, -1)
            cv2.grabCut(s, mask, None, None, None, 3, cv2.GC_INIT_WITH_MASK)

            la = labels.copy()
            la[np.where((mask == 0))] = 0
            labels_in_cir = np.unique(la)

            for i in labels_in_cir:
                l_piece = np.zeros_like(labels,
                                        dtype='uint8')  # labels(조각들) 사용해서 predicted image랑 비교   # numpy broadcasting 연산?
                l_piece[labels == i] = 255

                cut_p = cv2.bitwise_and(l_piece, mask)
                overlab_p = cv2.bitwise_and(l_piece, cut_p)

                overlap_area = cv2.countNonZero(overlab_p)
                p_area = cv2.countNonZero(l_piece)

                if overlap_area / p_area > 0.4:
                    mask2[labels == i] = 5
                else:
                    mask2[labels == i] = 8

            out[np.where((mask == 1) | (mask == 3))] = s[np.where((mask == 1) | (mask == 3))]
            out[np.where((mask2 == 5))] = s[np.where((mask2 == 5))]
            out[np.where((mask2 == 8))] = (255, 255, 255)
            sl = sl + [out]
        return sl

    def make_partial(self, colored_mandala):  # 그랩켯으로 레이어 별로 컬러이미지 생성
        pltt, md = colored_mandala[0], colored_mandala[1]
        partial = []
        sl = self.layered(md)
        for j in range(len(sl) - 1):
            tmp = sl[j].copy()
            tmp[np.where((md == (0, 0, 0)).all(axis=-1))] = md[np.where((md == (0, 0, 0)).all(axis=-1))]
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            partial.append(tmp)
        return pltt, partial

    def update(self, current_mandala, clen, mode='recommendation'):
        global WORKING

        pltt, layers = None, None
        if mode == 'recommendation':
            cv2.imwrite('current_mandala.jpg', current_mandala)
            current = img_as_ubyte(resize(cv2.imread('current_mandala.jpg'), (512, 512)))[:, :, :3]
            state, quant_mand = self.quantization(current)
            WORKING = 0.2
            self.user_mandala = self.fill_current_image(quant_mand)
            self.colors.colorlist = {}
            WORKING = 0.5
            self.colors.recommend_palette(clen, state, self.color_info)  ## 입력 개수 다양하게 들어갈 수 있도록 수정하기 ##
            self.colored_mandalas = self.mandala.color_partitions(current=self.user_mandala,
                                                                  colorlist=self.colors.colorlist[clen],
                                                                  mode=mode)
            WORKING = 0.7
            print("Generate", len(self.colored_mandalas), "palettes.")
            pltt, layers = self.make_partial(self.colored_mandalas[clen])
        elif mode == 'change_palette':
            pltt, layers = self.make_partial(self.colored_mandalas[clen])
        return pltt, layers


##만다라 UI 구현 클래스
class Screen:
    def __init__(self, input_img):
        # initialize game window, etc
        pygame.init()
        pygame.display.set_caption('Manshik')
        self.screen = pygame.display.set_mode((1850, 1050))  # 1200, 700
        self.clock = pygame.time.Clock()
        self.running = True
        self.font_name = FONT_NAME
        self.colors = Coloring()
        self.present_pltt = [[255, 255, 255]] * 6
        self.org_mandala = cv2.imread(input_img)
        self.upman = UpdateMandala(self.org_mandala)
        self.user_mand = None  # captured mandala
        self.mode = 0  # full / partial mode
        self.clen = 0  # color index
        self.layer = 0  # layer index
        self.layers = [self.upman.mandala.edge_image]  # [순차적으로 레이어만 칠해진 이미지 5개 배열형태로 저장]
        self.full_image = None  # 전체 모드 때문에 따로 빼놓음
        self.pal_rec_cnt = 0
        self.record_frame = np.zeros((10, 10, 3))

    def image_registration(self, ref_img, reg_img):
        ref_img = cv2.resize(ref_img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
        reg_img = cv2.resize(reg_img, dsize=(512, 512), interpolation=cv2.INTER_AREA)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        can_ref = cv2.Canny(ref_gray, 0, 175)
        can_ref = cv2.dilate(can_ref, kernel, iterations=2)

        reg_gray = cv2.cvtColor(reg_img, cv2.COLOR_BGR2GRAY)
        can_reg = cv2.Canny(reg_gray, 0, 175)
        can_reg = cv2.dilate(can_reg, kernel, iterations=2)

        height, width = reg_img.shape[:2]
        orb_detector = cv2.ORB_create(15000)

        kp1, d1 = orb_detector.detectAndCompute(can_reg, None)
        kp2, d2 = orb_detector.detectAndCompute(can_ref, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = matcher.match(d1, d2)
        matches.sort(key=lambda x: x.distance)

        matches = matches[:int(len(matches) * 90)]
        no_of_matches = len(matches)

        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))
        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        transformed_img = cv2.warpPerspective(reg_img,
                                              homography, (width, height), borderValue=(255, 255, 255))
        transformed_img[(transformed_img > 200).all(axis=-1)] = 255
        return transformed_img

    def frame2mandala(self, frame):
        heightImg, widthImg = 800, 1200

        image = cv2.imread(frame)
        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))  # split into HSV components

        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        nonSat = s < 50  # Find all pixels that are not very saturated

        # Slightly decrease the area of the non-satuared pixels by a erosion operation.
        nonSat = cv2.erode(nonSat.astype(np.uint8), disk)

        contours, _ = cv2.findContours(nonSat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
        biggest, _ = utils.biggestContour(contours)  # FIND THE BIGGEST CONTOUR

        if biggest.size != 0:
            biggest = utils.reorder(biggest)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(image, matrix, (widthImg, heightImg))

            crop = imgWarpColored[80:imgWarpColored.shape[0] - 90, 20:imgWarpColored.shape[1] - 500]

            crop = cv2.flip(crop, -1)
            cv2.imwrite(filename='usr_man/c_%d.jpg' % (self.pal_rec_cnt), img=crop)
            crop = cv2.imread('usr_man/c_%d.jpg' % (self.pal_rec_cnt))

            print('user mandala crop done')
            self.user_mand = self.image_registration(self.org_mandala, crop)
            return 0
        else:
            print('fail crop')
            return -1

    def draw_text(self, text, coord, size=30, color=(255, 255, 255)):
        font = pygame.font.Font(self.font_name, size)
        text_surface = font.render(text, True, color)
        loc = (coord[0], coord[1])
        self.screen.blit(text_surface, loc)

    def palette_update(self):
        # user palette
        bar_start = 0
        total_len = 720  # user palette length
        ratio_sum = sum([i for i, _ in self.upman.color_info])
        pygame.draw.rect(self.screen, (255, 255, 255), (1100, 130, total_len, 110), 4)  # user 틀

        for r, p in self.upman.color_info:
            bar_width = total_len * r * (1 / ratio_sum)
            pygame.draw.rect(self.screen, p[::-1], (1100 + bar_start, 130, bar_width, 110))
            bar_start += bar_width

        if total_len - bar_start != 0:  # 아직 덜칠한 비율만큼 흰색으로 채움
            pygame.draw.rect(self.screen, (255, 255, 255), (1100 + bar_start, 130, total_len - bar_start, 110))

        x = 0
        for p in self.present_pltt[:6]:
            pygame.draw.rect(self.screen, p[::-1], (1100 + x, 340, 110, 110))
            x += 120
        pygame.display.update()

    def pil2pyimage(self, img):
        # Convert PIL image to pygame surface image
        image = Image.fromarray(img)
        py_image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
        return py_image

    def adjust_color(self, image):
        img = Image.fromarray(image)
        im_out = ImageEnhance.Color(img).enhance(1.3)
        return np.array(im_out)

    def adjust_hsv(self, image, s_scale, v_scale):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsvImage = np.float32(hsvImage)

        H, S, V = cv2.split(hsvImage)  # 분리됨

        S = np.clip(S * s_scale, 0, S)  # 계산값, 최소값, 최대값
        V = np.clip(V * v_scale, 100, 255)  # 계산값, 최소값, 최대값

        hsvImage = cv2.merge([H, S, V])
        hsvImage = np.uint8(hsvImage)
        imgBgr = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

        return imgBgr

    def process_key_event(self, mode, image):
        self.mode = mode
        h, w = image.shape[:2]
        d_img = image[25:h - 25, 25:w - 25, :]

        if self.mode in [0, 1]:
            d_img = self.adjust_hsv(d_img, 1.5, 0.9)
            d_img = self.adjust_color(d_img)
            edge = self.upman.mandala.edge_image[25:h - 25, 25:w - 25, :]
            d_img[np.where((edge == (0, 0, 0)).all(axis=-1))] = (0, 0, 0)

        colored = self.pil2pyimage(d_img)
        self.show_main(input_img=colored)

    def capture(self, m):
        time.sleep(0.1)
        webcam = cv2.VideoCapture(0)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        check, frame = webcam.read()
        if m == 'recommendation':
            cv2.imwrite(filename='./cam_capture.jpg', img=frame)
            print('capture ok')
        if m == 'record':
            self.record_frame = frame
        webcam.release()
        # return frame

    def record_log(self, changed_mode, layer_index, key_input):
        global LOG_INDEX
        global LOG_DF
        LOG_INDEX += 1

        now = time.localtime()
        now_time = '%d : %d : %d' % (now.tm_hour, now.tm_min, now.tm_sec)

        new_log = [
            (now_time, changed_mode, self.pal_rec_cnt, '%d_%d' % (self.pal_rec_cnt, self.clen), layer_index, key_input)]
        new_log_df = pd.DataFrame(new_log, columns=LOG_DF.columns, index=[LOG_INDEX])
        LOG_DF = pd.concat([LOG_DF, new_log_df])

    def loadingbar(self):
        global WORKING

        t = threading.currentThread()
        FONT = pygame.font.Font('./PokemonGb-RAeo.ttf', 23)
        self.screen.fill(BGCOLOR)

        while getattr(t, "do_run", True):
            pygame.draw.rect(self.screen, (125, 125, 125), (1100, 600, 720, 110), 5)
            loading = FONT.render('Loading progress', True, (190, 0, 255))
            loading_rect = loading.get_rect(center=(1400, 550))
            self.screen.blit(loading, loading_rect)

            bar_width = WORKING * 720
            pygame.draw.rect(self.screen, (190, 0, 255), (1100, 602, bar_width, 100))

            pygame.display.update()
            self.clock.tick(60)

        WORKING = 0
        pygame.display.update()

    def wait_for_key(self):
        waiting = True
        global WORKING

        while waiting:
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if self.mode == 1:  # 전체 모드
                        print('entered')
                        if event.key == pygame.K_KP_ENTER:  # 다시 레이어모드로 돌아감
                            self.record_log(1, 5, 'ENTER')
                            waiting = False
                            self.process_key_event(0, self.layers[self.layer])

                        if event.key == KEY_0:  # 흰배경 전환
                            self.record_log(2, 5, 0)
                            waiting = False
                            img = cv2.imread('blackback.jpg')
                            self.process_key_event(3, img)

                        if event.key == KEY_8:  # 다음 팔레트로
                            if self.clen == len(self.upman.colored_mandalas) - 1:
                                print('Last palette.')
                                pyautogui.alert('Last palette.')
                                self.process_key_event(self.mode, self.full_image)

                            self.clen += 1
                            print('next palette, ID:', self.clen)
                            self.present_pltt, self.layers = self.upman.update(self.user_mand, self.clen,
                                                                               mode='change_palette')
                            self.full_image = self.layers[-1]

                            waiting = False
                            self.record_log(self.mode, 5, 8)
                            self.process_key_event(self.mode, self.full_image)

                        if event.key == KEY_2:  # 이전 팔레트로
                            self.clen -= 1
                            if self.clen < 0:  # 팔레트 처음 색일 경우
                                self.clen = 0
                                print('First palette.')
                                pyautogui.alert('First palette.')
                                self.process_key_event(self.mode, self.full_image)

                            print('prior palette, ID:', self.clen)
                            self.present_pltt, self.layers = self.upman.update(self.user_mand, self.clen,
                                                                               mode='change_palette')
                            self.full_image = self.layers[-1]

                            waiting = False
                            self.record_log(self.mode, 5, 2)
                            self.process_key_event(self.mode, self.full_image)

                        if event.key == KEY_5:  # 색상추천 5번키 & 카메라 처리
                            os.makedirs('./recommended', exist_ok=True)
                            os.makedirs('./usr_man', exist_ok=True)

                            print('generating palette sets...')
                            self.screen.fill(BGCOLOR)
                            pygame.display.update()

                            self.capture(m='recommendation')
                            time.sleep(0.1)
                            self.clen = 0
                            self.pal_rec_cnt += 1
                            frame = "cam_capture.jpg"
                            flag = self.frame2mandala(frame)

                            if flag == -1:
                                pyautogui.alert('Cropping Failed.')
                                continue

                            t = threading.Thread(target=self.loadingbar, args=())
                            t.start()
                            pygame.display.update()

                            self.present_pltt, self.layers = self.upman.update(self.user_mand, self.clen,
                                                                               mode='recommendation')
                            WORKING = 0.8
                            self.full_image = self.layers[-1]

                            for i, im in enumerate(self.upman.colored_mandalas):  # recommended 만다라 전부 저장
                                cv2.imwrite(filename='recommended/%d_%d.jpg' % (self.pal_rec_cnt, i),
                                            img=np.array(im[1]))

                            cv2.imwrite(filename='usr_man/%d.jpg' % (self.pal_rec_cnt), img=self.user_mand)

                            self.record_log(self.mode, 5, 5)
                            WORKING = 1.0
                            time.sleep(0.5)
                            waiting = False
                            t.do_run = False
                            self.process_key_event(self.mode, self.full_image)

                    if self.mode == 2:  # 백지모드-1
                        print('black')
                        if event.key == KEY_0:
                            self.record_log(0, self.layer, 0)
                            waiting = False
                            self.process_key_event(0, self.layers[self.layer])

                    if self.mode == 3:  # 백지모드-2
                        print('black')
                        if event.key == KEY_0:
                            self.record_log(1, 5, 0)
                            waiting = False
                            self.process_key_event(1, self.full_image)

                    if self.mode == 0:  # 레이어 모드
                        if event.key == pygame.K_KP_ENTER:  # 전체모드로 바뀜
                            self.record_log(1, 5, 'ENTER')
                            waiting = False
                            self.process_key_event(1, self.full_image)

                        if event.key == KEY_0:  # 백지모드
                            self.record_log(2, self.layer, 0)
                            waiting = False
                            img = cv2.imread('blackback.jpg')
                            self.process_key_event(2, img)

                        if self.layers != 0:
                            if event.key == KEY_4:  # 이전 레이어 보여줌
                                print('prior')
                                if self.layer < 1:
                                    print('first_layer')
                                    self.record_log(0, self.layer, 4)
                                    self.process_key_event(self.mode, self.layers[self.layer])
                                self.layer -= 1
                                self.record_log(0, self.layer, 4)
                                waiting = False
                                self.process_key_event(self.mode, self.layers[self.layer])

                            if event.key == KEY_6:  # 다음레이어 보여줌
                                print('next')
                                if self.layer > 4:
                                    print('last_layer')
                                    self.process_key_event(self.mode, self.layers[self.layer])
                                    self.record_log(0, self.layer, 6)
                                self.layer += 1
                                self.record_log(0, self.layer, 6)
                                waiting = False
                                self.process_key_event(self.mode, self.layers[self.layer])

                        if event.key == KEY_8:  # 다음 팔레트로
                            if self.clen == len(self.upman.colored_mandalas) - 1:
                                print('Last palette.')
                                pyautogui.alert('Last palette.')
                                self.process_key_event(self.mode, self.layers[self.layer])

                            self.clen += 1
                            print('next palette, ID:', self.clen)
                            self.present_pltt, self.layers = self.upman.update(self.user_mand, self.clen,
                                                                               mode='change_palette')
                            self.full_image = self.layers[-1]

                            waiting = False
                            self.record_log(self.mode, self.layer, 8)
                            self.process_key_event(self.mode, self.layers[self.layer])

                        if event.key == KEY_2:  # 이전 팔레트로
                            self.clen -= 1
                            if self.clen < 0:  # 팔레트 처음 색일 경우
                                self.clen = 0
                                print('First palette.')
                                pyautogui.alert('First palette.')
                                self.process_key_event(self.mode, self.layers[self.layer])

                            print('prior palette, ID:', self.clen)
                            self.present_pltt, self.layers = self.upman.update(self.user_mand, self.clen,
                                                                               mode='change_palette')
                            self.full_image = self.layers[-1]

                            waiting = False
                            self.record_log(self.mode, self.layer, 2)
                            self.process_key_event(self.mode, self.layers[self.layer])

                        if event.key == KEY_5:  # 색상추천 5번키 & 카메라 처리
                            os.makedirs('./recommended', exist_ok=True)
                            os.makedirs('./usr_man', exist_ok=True)

                            print('generating palette sets...')
                            self.screen.fill(BGCOLOR)
                            pygame.display.update()

                            self.capture(m='recommendation')
                            time.sleep(0.5)
                            self.clen = 0
                            self.pal_rec_cnt += 1
                            frame = "cam_capture.jpg"
                            flag = self.frame2mandala(frame)

                            if flag == -1:
                                pyautogui.alert('Cropping Failed.')
                                self.wait_for_key()
                                continue

                            t = threading.Thread(target=self.loadingbar, args=())
                            t.start()
                            pygame.display.update()

                            self.present_pltt, self.layers = self.upman.update(self.user_mand, self.clen,
                                                                               mode='recommendation')
                            WORKING = 0.8
                            self.full_image = self.layers[-1]

                            for i, im in enumerate(self.upman.colored_mandalas):  # recommended 만다라 전부 저장
                                cv2.imwrite(filename='recommended/%d_%d.jpg' % (self.pal_rec_cnt, i),
                                            img=np.array(im[1]))

                            cv2.imwrite(filename='usr_man/%d.jpg' % (self.pal_rec_cnt), img=self.user_mand)

                            self.record_log(self.mode, self.layer, 5)
                            WORKING = 1.0
                            time.sleep(0.5)
                            waiting = False
                            t.do_run = False
                            self.process_key_event(self.mode, self.layers[self.layer])

                if event.type == pygame.QUIT:
                    LOG_DF.to_csv('usr_log.csv', mode='w', index=False)
                    waiting = False
                    self.running = False
                    print('Quit the game...')
                    pygame.quit()
                    sys.exit()

    def show_main(self, input_img):
        self.screen.fill(BGCOLOR)

        base_img = input_img
        print('mode:', self.mode)
        if self.mode == 0:  # 레이어모드
            self.draw_text("LAYER MODE", coord=(1100, 980))

        elif self.mode == 1:  # 전체화면 모드
            self.draw_text("FULLY COLORED MODE", coord=(1100, 980))

        elif self.mode == 2 or self.mode == 3:  # 백지모드
            self.draw_text("BLACK BOARD MODE", coord=(1100, 980))

        self.draw_text("YOUR PALLETE", size=25, coord=(1100, 70))
        self.draw_text("AI PALLETE", size=25, coord=(1100, 200 * 1.5))
        self.draw_text("My MANDALA", size=25, coord=(1100, 500))

        if os.path.isfile('current_mandala.jpg.'):
            user_img = pygame.image.load('current_mandala.jpg.')
            user_img = pygame.transform.scale(user_img, (400, 400))
            self.screen.blit(user_img, [1100, 550])

        pre_img = pygame.transform.scale(base_img, (1000, 1000))
        self.screen.blit(pre_img, [55, 30])

        self.palette_update()
        pygame.display.update()
        self.wait_for_key()


def main():
    input_img = 'crop_image/111_org.jpg'
    g = Screen(input_img)
    try:
        while g.running:
            g.show_main(input_img=pygame.image.load(input_img))
    except:
        LOG_DF.to_csv('usr_log.csv', index=False, mode='w')
        pygame.quit()
        sys.exit()


main()