# なぜかOpenAiGymのデフォルトのビデオ保存機能が上手く行かないので自作
from matplotlib import animation
import matplotlib.pyplot as plt

class GymVideo(object):
  def __init__(self):
    self.frames_ = []

  def append_frame(self,frame):
    self.frames_.append(frame)

  def save_video(self, path):
    frame = self.frames_[0]
    fig = plt.figure(figsize = (frame.shape[1] // 100, frame.shape[0] // 100))

    patch = plt.imshow(self.frames_[0])

    def animate(i):
      patch.set_data(self.frames_[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(self.frames_), interval=10)
    anim.save(path, writer = 'Pillow')

    self.frames_ = []
    plt.clf()
    plt.close()
  
  def __len__(self):
    return len(self.frames_)

