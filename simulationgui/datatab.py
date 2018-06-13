
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton


class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.pushButton1 = QPushButton("PyQt button")
        self.layout.addWidget(self.pushButton1)
        self.setLayout(self.layout)
