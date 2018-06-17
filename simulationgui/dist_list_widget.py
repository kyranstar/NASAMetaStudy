# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dist_list_widget.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DistListItem(object):
    def setupUi(self, DistListItem):
        DistListItem.setObjectName("DistListItem")
        DistListItem.resize(516, 300)
        self.horizontalLayout = QtWidgets.QHBoxLayout(DistListItem)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.name_input = QtWidgets.QLineEdit(DistListItem)
        self.name_input.setObjectName("name_input")
        self.horizontalLayout.addWidget(self.name_input)
        self.type_combo = QtWidgets.QComboBox(DistListItem)
        self.type_combo.setObjectName("type_combo")
        self.type_combo.addItem("")
        self.type_combo.addItem("")
        self.horizontalLayout.addWidget(self.type_combo)
        self.variance_label = QtWidgets.QLabel(DistListItem)
        self.variance_label.setObjectName("variance_label")
        self.horizontalLayout.addWidget(self.variance_label)
        self.variance_input = QtWidgets.QLineEdit(DistListItem)
        self.variance_input.setObjectName("variance_input")
        self.horizontalLayout.addWidget(self.variance_input)
        self.mean_label = QtWidgets.QLabel(DistListItem)
        self.mean_label.setObjectName("mean_label")
        self.horizontalLayout.addWidget(self.mean_label)
        self.mean_input = QtWidgets.QLineEdit(DistListItem)
        self.mean_input.setObjectName("mean_input")
        self.horizontalLayout.addWidget(self.mean_input)
        self.delete_button = QtWidgets.QPushButton(DistListItem)
        self.delete_button.setObjectName("delete_button")
        self.horizontalLayout.addWidget(self.delete_button)
        self.variance_label.setBuddy(self.variance_input)
        self.mean_label.setBuddy(self.mean_input)

        self.retranslateUi(DistListItem)
        QtCore.QMetaObject.connectSlotsByName(DistListItem)

    def retranslateUi(self, DistListItem):
        _translate = QtCore.QCoreApplication.translate
        DistListItem.setWindowTitle(_translate("DistListItem", "Form"))
        self.name_input.setText(_translate("DistListItem", "Name"))
        self.type_combo.setItemText(0, _translate("DistListItem", "Normal"))
        self.type_combo.setItemText(1, _translate("DistListItem", "Categorical"))
        self.variance_label.setText(_translate("DistListItem", "Variance:"))
        self.variance_input.setText(_translate("DistListItem", "0.0"))
        self.mean_label.setText(_translate("DistListItem", "Mean:"))
        self.mean_input.setText(_translate("DistListItem", "0.0"))
        self.delete_button.setText(_translate("DistListItem", "Delete"))

