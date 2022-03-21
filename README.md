# Calango Project

It looks like magic

![Tests](https://github.com/cereja-project/calango/workflows/Python%20package/badge.svg)
![PyPi Publish](https://github.com/cereja-project/calango/workflows/Upload%20Python%20Package/badge.svg)
[![PyPI version](https://badge.fury.io/py/calango.svg)](https://badge.fury.io/py/calango)

## Get started

Install with pip

`pip install calango`

or

`python -m pip install calango`

## Mouse Interface

```python
from calango.devices import Mouse

mouse = Mouse()

mouse.up()  # move mouse pointer [up, down, left, right, top_left, top_right ...

mouse.position = (10, 10)  # move mouse pointer to (x, y)
print(mouse.position)  # return current postion

```
## Image API
### Image reading

```python
from calango import Image

image = Image(image_or_path='image.png')
image[:50, :50].plot() # plot image cropped (0:50, 0:50)
image.right.top.plot() # get image cropped at right top corner
image.draw_text('Hello World!', pos='left_bottom')  # draw text on image
image.height  # return image height
image.width  # return image width
image.center_position  # return image center position
image.crop_by_center((20, 20))  # crop image by center
image.prune()  # resize image to square size by min(width, height)
# ... and more
```

## Video API
### Video Writer
```python
from calango import VideoWriter
import numpy as np

frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
with VideoWriter('test.mp4', fps=10) as writer:
    for frame in frames:
        writer.add_frame(frame)
# OR INLINE
VideoWriter.write_frames('test.mp4', frames) # on end is closed automatically
```
### Video from Camera
```python
from calango import Video

cam = Video(0) # 0 is the default camera
cam.show() # is running in a new thread

cam.total_frames # return the total number of frames
cam.fps # return the frames per second
cam.is_opened # return True if the camera is opened
# ... and more
```

### Video from File
```python
from calango import Video

cam = Video('./video.mp4')
cam.show()
```
### Video from frames directory
```python
from calango import Video

cam = Video('./video/', fps=24)
cam.show()
```
### Video from sequence of images
```python
from calango import Video

images = ['./video/img1.jpg', './video/img2.jpg', './video/img3.jpg'] # or list of images
cam = Video(images, fps=10)
cam.show()
```