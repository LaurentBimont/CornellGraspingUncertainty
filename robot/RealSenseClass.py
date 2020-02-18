import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

class RealCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.frame = None
        self.pipelineStarted = False
        self.depth = None
        self.rgb = None
        self.depth_scale = None
        self.align = None
        self.color_image = None
        self.depth_image = None

    def start_pipe(self, align=True, usb3=True):
        if not self.pipelineStarted:
            if align:
                # Create a config and configure the pipeline to stream
                #  different resolutions of color and depth streams
                config = rs.config()

                #if usb3:
                 #   config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
                #    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

                #else:
                self.profile = config.resolve(self.pipeline)  # does not start streaming

             #   self.profile = self.pipeline.start(config)
                self.pipelineStarted = True

                #align the two streams
                align_to = rs.stream.color
                self.align = rs.align(align_to)

                # Get depth scale
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()

    def stop_pipe(self):
        if self.pipelineStarted:
            self.pipeline.stop()
            self.pipelineStarted = False

    def show(self):
        plt.subplot(1, 2, 1)
        plt.imshow(self.depth_image)
        plt.subplot(1, 2, 2)
        plt.imshow(self.color_image)
        plt.show()

    def get_frame(self):
        # Get frameset of color and depth
        self.frame = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = self.align.process(self.frame)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        self.depth_image = np.asanyarray(aligned_depth_frame.get_data())*self.depth_scale
        self.color_image = np.asanyarray(color_frame.get_data())
        return self.depth_image, self.color_image


if __name__=='__main__':
    Cam = RealCamera()
    Cam.start_pipe(usb3=False)
    Cam.get_frame()
    Cam.show()
    Cam.stop_pipe()
