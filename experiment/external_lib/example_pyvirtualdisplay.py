# fmt:off
from pyvirtualdisplay import Display
display = Display(visible=False, backend="xvfb")
display.start()
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position
# fmt:on
if __name__ == "__main__":
  plt.plot([1, 2, 3, 4])
  plt.show()
  plt.close()

display.stop()
