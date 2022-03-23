import base64
import io
from .settings import ON_COLAB_JUPYTER

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
