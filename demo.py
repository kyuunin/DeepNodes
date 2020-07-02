from Qt import QtCore, QtWidgets
from DeepNodes import *
from Nodz import *

try:
    app = QtWidgets.QApplication([])
except:
    # I guess we're running somewhere that already has a QApp created
    app = None

nodz = nodz_main.Nodz(None)
# nodz.loadConfig(filePath='')
nodz.initialize()
nodz.show()


if app:
    # command line stand alone test... run our own event loop
    app.exec_()

