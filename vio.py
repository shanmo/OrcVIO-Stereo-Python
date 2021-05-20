
from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from orcvio import ORCVIO
import utils 


class VIO(object):
    def __init__(self, config, img_queue, imu_queue, gt_queue, viewer=None):
        
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.gt_queue = gt_queue

        self.image_processor = ImageProcessor(config)
        self.orcvio = ORCVIO(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.gt_thread = Thread(target=self.process_gt)

        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()
        self.gt_thread.start()

        self.first_gt_flag = False 

    def process_gt(self):

        while True:
            x = self.gt_queue.get()
            if x is None:
                return

            data = x

            if not self.first_gt_flag:
                p0 = data.p
                self.first_gt_flag = True 

            R = utils.to_rotation(data.q) 
            gt_pose = utils.Isometry3d(R, data.p - p0)

            if self.viewer is not None:
                self.viewer.update_gt(gt_pose)

    def process_img(self):

        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            # print('img_msg', img_msg.timestamp)

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stereo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):

        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return
            # print('imu_msg', imu_msg.timestamp)

            self.image_processor.imu_callback(imu_msg)
            self.orcvio.imu_callback(imu_msg)

    def process_feature(self):

        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return
            # print('feature_msg', feature_msg.timestamp)
            result = self.orcvio.feature_callback(feature_msg)

            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)
        


if __name__ == '__main__':

    import time

    from dataset import EuRoCDataset, DataPublisher
    from viewer import Viewer

    dataset_path = "/media/erl/disk2/euroc/MH_01_easy"

    viewer = Viewer()

    dataset = EuRoCDataset(dataset_path)
    offset = 40
    dataset.set_starttime(offset)   # start from static state

    img_queue = Queue()
    imu_queue = Queue()
    gt_queue = Queue()

    config = ConfigEuRoC()
    Orc_VIO = VIO(config, img_queue, imu_queue, gt_queue, viewer=viewer)

    duration = float('inf')
    # make it smaller if image processing and OrcVIO computation is slow
    if not config.load_features_flag:
        ratio = 1.0
    else:
        ratio = 0.4
        
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    # publish stereo image 
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)
    gt_publisher = DataPublisher(
        dataset.groundtruth, gt_queue, duration)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
    gt_publisher.start(now)