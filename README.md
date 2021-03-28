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

## Camera Interface

```python
from calango.devices import Capture
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
cv2.imwrite('calango.png', frame)
cam.stop()  # stop capture

```