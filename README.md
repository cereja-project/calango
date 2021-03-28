# magic

It looks like magic

![Tests](https://github.com/cereja-project/magic/workflows/Python%20Tests/badge.svg)
![PyPi Publish](https://github.com/cereja-project/magic/workflows/PyPi%20Publish/badge.svg)
[![PyPI version](https://badge.fury.io/py/magic.svg)](https://badge.fury.io/py/magic)

## Get started

Install with pip

`pip install magic`

or

`python -m pip install magic`

## Mouse Interface

```python
from magic.devices import VideoCapture, Mouse

mouse = Mouse()

mouse.up()  # move mouse pointer [up, down, left, right, top_left, top_right ...

mouse.position = (10, 10)  # move mouse pointer to (x, y)
print(mouse.position)  # return current postion

```

## Camera Interface

```python
from magic.devices import Capture
import cv2

cam = Capture()
while True:
    frame = next(cam)

    cv2.imshow('Window Name', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cam.stop()

# You can use out while context
frame = cam.frame  # current frame numpy array
cv2.imwrite('magic.png', frame)
cam.stop()  # stop capture

```