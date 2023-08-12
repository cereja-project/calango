import base64
import io
from .settings import ON_COLAB_JUPYTER
from matplotlib import pyplot as plt
import math

__all__ = ['show_local_mp4']


def show_local_mp4(file_name, width=640, height=480):
    assert ON_COLAB_JUPYTER, 'show_local_mp4 can bee used only on colab runtime'
    # noinspection PyUnresolvedReferences
    from IPython.display import HTML
    video_encoded = base64.b64encode(io.open(file_name, "rb").read())
    return HTML(
            data="""<video width="{0}" height="{1}" alt="test" controls>
                  <source src="data:video/mp4;base64,{2}" type="video/mp4" />
                </video>""".format(
                    width, height, video_encoded.decode("ascii")
            )
    )


class ImagePlot:
    def __init__(self, fig_size=(13, 13), max_cols=4):
        self.fig_size = fig_size
        self.figure = plt.figure(figsize=fig_size)  # TODO: does it need to be global?
        self._max_cols = max_cols
        self._total_images = 0

    @property
    def current_row(self):
        return math.ceil(self._total_images / self._max_cols)

    @property
    def current_col(self):
        return ((self._total_images - 1) % self._max_cols) + 1

    def plot(self, image, title=None, color='gray'):
        try:
            self._total_images += 1
            plt.subplot(self._max_cols, self._max_cols, self._total_images)
            plt.title(self.current_col if title is None else title)
            plt.xticks([]), plt.yticks([])
            plt.imshow(image.copy(), color)
        except Exception as err:
            self._total_images -= 1  # garante consistencia da quantidade de imagens
            raise err
