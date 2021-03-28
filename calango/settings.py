import os

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'
try:
    # noinspection PyUnresolvedReferences
    IPYTHON = get_ipython()
    ON_COLAB_JUPYTER = True if "google.colab" in IPYTHON.__str__() else False
except NameError:
    IPYTHON = None
    ON_COLAB_JUPYTER = False
