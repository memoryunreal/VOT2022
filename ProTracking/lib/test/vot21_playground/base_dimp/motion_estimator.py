import cv2
import numpy as np


class MotionEstimator(object):
    """ Statistic of the track """
    def __init__(self):
        self.pos = None
        self.vel = None
        self.conf = 0
        self.vel_his = []  # velocity history
        self.vel_his_conf = []  # velocity history

    def initialize(self, box):
        self.pos = box[:2] + box[2:]/2
        self.wh = box[2:]
        self.vel_his = []  # velocity history
        self.vel_his_conf = []  # velocity history

    def update_vel_his(self, v, conf):
        self.vel_his.append(v)
        self.vel_his_conf.append(conf)
        if len(self.vel_his) > 20:
            self.vel_his.pop(0)
            self.vel_his_conf.pop(0)
        vel_his = np.stack(self.vel_his)
        vel_conf = np.stack(self.vel_his_conf)
        vel_conf_norm = vel_conf/vel_conf.sum()

        vel_mean = vel_conf_norm.dot(vel_his)
        vel_var = vel_his.var(0)
        return vel_mean, vel_var

    def cosine_pdf(self, x, x_, size):
        """
        Args:
            x: Tracker预测位置
            x_: 运动模型预测位置
            size: 目标尺寸
        """
        PI = 3.1415
        phase = min(np.linalg.norm(x-x_)/size, 1)*PI/2
        return np.cos(phase)

    def update(self, box, conf, th=0.3):
        pos = box[:2] + box[2:]/2
        if conf > th:
            self.wh = box[2:]

        _v = pos - self.pos  # 瞬时速度
        if self.vel is None:
            self.vel = _v  # 初始化estimator的速度
        vel_mean, vel_var = self.update_vel_his(_v, conf)  # 目标运动速度的统计量

        self.vel = vel_mean
        self.pos = pos
        self.conf = conf
        return box, self.conf

    @property
    def ltwh(self):
        return np.concatenate([self.pos - self.wh/2, self.wh])

    def rescore(self, box, conf):
        pos = box[:2] + box[2:]/2

        _v = pos - self.pos  # 瞬时速度
        if self.vel is None:
            self.vel = _v  # 初始化estimator的速度

        motion_conf = self.cosine_pdf(pos, self.pos + self.vel, size=np.sqrt(self.wh.prod()))  # 当前目标运动速度的可信度
        print(motion_conf)
        conf = conf * motion_conf
        return conf

    def predict(self):
        """ 如果Tracker置信度不佳, 则依赖Motion Model估计目标位置 """
        self.pos = self.pos + self.vel
        self.vel = self.vel * 0.8  # 认为消失的目标不会一直按照原趋势运动
        self.conf = self.conf * 0.8  # 不认为被遮挡后的目标特别值得相信
        return self.ltwh, self.conf


def imshow(img, win_name=''):
    cv2.imshow(win_name, img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit(0)


def draw_box(img, box, format='xywh'):
    if format == 'xywh':
        return cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255), 2)
    elif format == 'ltbr':
        return cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    else:
        raise NotImplementedError


class Cat(object):
    """ Pretending it is a cat straying """
    def __init__(self, box, vel):
        """
        Args:
            box: (`np.ndarray`) in [x, y, w, h]
        """
        self.wh = box[2:]
        self.pos = box[:2] + self.wh/2  # center positaion of the cat
        self.vel = vel  # velocity of the cat
        self.conf = 1.0

    def cat_mood(self):
        return np.random.rand(1)

    @property
    def ltwh(self):
        return np.concatenate([self.pos - self.wh/2, self.wh])

    def pace(self):
        if self.cat_mood() > 0.9:
            _v = 50 * (np.random.rand(2)-0.5) * np.array([5, 5], dtype=np.float)
            _pos = self.pos + _v
            conf = 1 - (np.linalg.norm(_v) / np.sqrt(np.prod(self.wh))) ** 2
            return np.concatenate([_pos - self.wh / 2, self.wh]), max(0, conf)

        self.pos += self.vel

        vel_noise = 0.3*(np.random.rand(2)-0.5)
        self.vel = self.vel + vel_noise

        # if self.vel.max().__abs__() > 10:
        #     print(self.vel)
        return np.concatenate([self.pos-self.wh/2, self.wh]), self.conf


if __name__ == '__main__':
    W, H = 1280, 720
    cat = Cat(np.array([200, 300, 70, 80], dtype=np.float32), np.array([5, 0], dtype=np.float32))

    me = MotionEstimator()
    me.initialize(cat.ltwh)

    while True:
        canvas = 255 * np.ones([H, W]).astype(np.uint8)
        box, conf = cat.pace()
        print(conf)
        conf = me.rescore(box, conf)
        print(conf)
        print()
        if conf < 0.1:
            box, conf = me.predict()
        else:
            box, conf = me.update(box, conf)
        # print(box, conf)
        draw_box(canvas, box)
        imshow(canvas)
        if box[0] < 0 or box[0] > W or box[1] < 0 or box[1] > H:
            cat = Cat(np.array([200, 300, 70, 80], dtype=np.float32), np.array([5, 0], dtype=np.float32))
