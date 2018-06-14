# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(952, 1585)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tab_panel = QtWidgets.QTabWidget(self.centralwidget)
        self.tab_panel.setObjectName("tab_panel")
        self.datatab = QtWidgets.QWidget()
        self.datatab.setObjectName("datatab")
        self.gridLayout = QtWidgets.QGridLayout(self.datatab)
        self.gridLayout.setObjectName("gridLayout")
        self.datatab_scroll = QtWidgets.QScrollArea(self.datatab)
        self.datatab_scroll.setMinimumSize(QtCore.QSize(0, 0))
        self.datatab_scroll.setWidgetResizable(True)
        self.datatab_scroll.setObjectName("datatab_scroll")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 900, 1459))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.distributions_label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.distributions_label.setObjectName("distributions_label")
        self.gridLayout_2.addWidget(self.distributions_label, 1, 0, 1, 1)
        self.distributions_list = DistributionsList(self.scrollAreaWidgetContents_2)
        self.distributions_list.setObjectName("distributions_list")
        self.gridLayout_2.addWidget(self.distributions_list, 2, 0, 1, 2)
        self.correlations_label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.correlations_label.setObjectName("correlations_label")
        self.gridLayout_2.addWidget(self.correlations_label, 5, 0, 1, 1)
        self.upload_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.upload_button.setObjectName("upload_button")
        self.gridLayout_2.addWidget(self.upload_button, 0, 0, 1, 4)
        self.true_model_panel = QtWidgets.QTextEdit(self.scrollAreaWidgetContents_2)
        self.true_model_panel.setMaximumSize(QtCore.QSize(270, 16777215))
        self.true_model_panel.setObjectName("true_model_panel")
        self.gridLayout_2.addWidget(self.true_model_panel, 2, 2, 1, 1)
        self.correlations_table = CorrelationsTable(self.scrollAreaWidgetContents_2)
        self.correlations_table.setObjectName("correlations_table")
        self.gridLayout_2.addWidget(self.correlations_table, 6, 0, 2, 3)
        self.add_var_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.add_var_button.setObjectName("add_var_button")
        self.gridLayout_2.addWidget(self.add_var_button, 3, 0, 1, 3)
        self.fit_distributions_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.fit_distributions_button.setObjectName("fit_distributions_button")
        self.gridLayout_2.addWidget(self.fit_distributions_button, 4, 0, 1, 3)
        self.true_model_label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.true_model_label.setObjectName("true_model_label")
        self.gridLayout_2.addWidget(self.true_model_label, 1, 2, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_2 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.lasso_cv_true_model = QtWidgets.QRadioButton(self.groupBox_2)
        self.lasso_cv_true_model.setObjectName("lasso_cv_true_model")
        self.verticalLayout_5.addWidget(self.lasso_cv_true_model)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.fit_true_model_button = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.fit_true_model_button.setObjectName("fit_true_model_button")
        self.verticalLayout_4.addWidget(self.fit_true_model_button)
        self.gridLayout_2.addLayout(self.verticalLayout_4, 2, 3, 1, 1)
        self.datatab_scroll.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout.addWidget(self.datatab_scroll, 0, 0, 1, 1)
        self.tab_panel.addTab(self.datatab, "")
        self.experimenttab = QtWidgets.QWidget()
        self.experimenttab.setObjectName("experimenttab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.experimenttab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.num_samples_max_label = QtWidgets.QLabel(self.experimenttab)
        self.num_samples_max_label.setObjectName("num_samples_max_label")
        self.gridLayout_3.addWidget(self.num_samples_max_label, 3, 2, 1, 1)
        self.num_samples_max_input = QtWidgets.QLineEdit(self.experimenttab)
        self.num_samples_max_input.setObjectName("num_samples_max_input")
        self.gridLayout_3.addWidget(self.num_samples_max_input, 4, 2, 1, 1)
        self.num_trials_input = QtWidgets.QLineEdit(self.experimenttab)
        self.num_trials_input.setObjectName("num_trials_input")
        self.gridLayout_3.addWidget(self.num_trials_input, 4, 0, 1, 1)
        self.num_samples_min_input = QtWidgets.QLineEdit(self.experimenttab)
        self.num_samples_min_input.setObjectName("num_samples_min_input")
        self.gridLayout_3.addWidget(self.num_samples_min_input, 4, 1, 1, 1)
        self.num_samples_min_label = QtWidgets.QLabel(self.experimenttab)
        self.num_samples_min_label.setObjectName("num_samples_min_label")
        self.gridLayout_3.addWidget(self.num_samples_min_label, 3, 1, 1, 1)
        self.num_samples_step_label = QtWidgets.QLabel(self.experimenttab)
        self.num_samples_step_label.setObjectName("num_samples_step_label")
        self.gridLayout_3.addWidget(self.num_samples_step_label, 3, 3, 1, 1)
        self.num_samples_step_input = QtWidgets.QLineEdit(self.experimenttab)
        self.num_samples_step_input.setObjectName("num_samples_step_input")
        self.gridLayout_3.addWidget(self.num_samples_step_input, 4, 3, 1, 1)
        self.num_trials_label = QtWidgets.QLabel(self.experimenttab)
        self.num_trials_label.setObjectName("num_trials_label")
        self.gridLayout_3.addWidget(self.num_trials_label, 3, 0, 1, 1)
        self.run_experiment_button = QtWidgets.QPushButton(self.experimenttab)
        self.run_experiment_button.setObjectName("run_experiment_button")
        self.gridLayout_3.addWidget(self.run_experiment_button, 4, 4, 1, 1)
        self.exp_controls = QtWidgets.QGridLayout()
        self.exp_controls.setObjectName("exp_controls")
        self.error_controls = QtWidgets.QVBoxLayout()
        self.error_controls.setObjectName("error_controls")
        self.groupBox = QtWidgets.QGroupBox(self.experimenttab)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.prediction_mse_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.prediction_mse_checkbox.setObjectName("prediction_mse_checkbox")
        self.verticalLayout_3.addWidget(self.prediction_mse_checkbox)
        self.matthews_coef_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.matthews_coef_checkbox.setObjectName("matthews_coef_checkbox")
        self.verticalLayout_3.addWidget(self.matthews_coef_checkbox)
        self.error_controls.addWidget(self.groupBox)
        self.exp_controls.addLayout(self.error_controls, 1, 0, 1, 1)
        self.var_subset_controls = QtWidgets.QVBoxLayout()
        self.var_subset_controls.setObjectName("var_subset_controls")
        self.selection_methods_group = QtWidgets.QGroupBox(self.experimenttab)
        self.selection_methods_group.setObjectName("selection_methods_group")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.selection_methods_group)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lasso_cv_checkbox = QtWidgets.QCheckBox(self.selection_methods_group)
        self.lasso_cv_checkbox.setObjectName("lasso_cv_checkbox")
        self.verticalLayout.addWidget(self.lasso_cv_checkbox)
        self.lasso_cv_std_checkbox = QtWidgets.QCheckBox(self.selection_methods_group)
        self.lasso_cv_std_checkbox.setObjectName("lasso_cv_std_checkbox")
        self.verticalLayout.addWidget(self.lasso_cv_std_checkbox)
        self.lasso_bic_checkbox = QtWidgets.QCheckBox(self.selection_methods_group)
        self.lasso_bic_checkbox.setObjectName("lasso_bic_checkbox")
        self.verticalLayout.addWidget(self.lasso_bic_checkbox)
        self.var_subset_controls.addWidget(self.selection_methods_group)
        self.graphics_metrics_group = QtWidgets.QGroupBox(self.experimenttab)
        self.graphics_metrics_group.setObjectName("graphics_metrics_group")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.graphics_metrics_group)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.perfectly_chosen_checkbox = QtWidgets.QCheckBox(self.graphics_metrics_group)
        self.perfectly_chosen_checkbox.setObjectName("perfectly_chosen_checkbox")
        self.verticalLayout_2.addWidget(self.perfectly_chosen_checkbox)
        self.predictors_missed_checkbox = QtWidgets.QCheckBox(self.graphics_metrics_group)
        self.predictors_missed_checkbox.setObjectName("predictors_missed_checkbox")
        self.verticalLayout_2.addWidget(self.predictors_missed_checkbox)
        self.false_predictors_checkbox = QtWidgets.QCheckBox(self.graphics_metrics_group)
        self.false_predictors_checkbox.setObjectName("false_predictors_checkbox")
        self.verticalLayout_2.addWidget(self.false_predictors_checkbox)
        self.symm_diff_checkbox = QtWidgets.QCheckBox(self.graphics_metrics_group)
        self.symm_diff_checkbox.setObjectName("symm_diff_checkbox")
        self.verticalLayout_2.addWidget(self.symm_diff_checkbox)
        self.var_subset_controls.addWidget(self.graphics_metrics_group)
        self.exp_controls.addLayout(self.var_subset_controls, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.exp_controls, 0, 4, 1, 1)
        self.exp_graph_scroll = QtWidgets.QScrollArea(self.experimenttab)
        self.exp_graph_scroll.setWidgetResizable(True)
        self.exp_graph_scroll.setObjectName("exp_graph_scroll")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 711, 1401))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.variable_subset_graph = QtWidgets.QGraphicsView(self.scrollAreaWidgetContents)
        self.variable_subset_graph.setMinimumSize(QtCore.QSize(0, 400))
        self.variable_subset_graph.setObjectName("variable_subset_graph")
        self.gridLayout_4.addWidget(self.variable_subset_graph, 1, 0, 1, 1)
        self.error_graph = QtWidgets.QGraphicsView(self.scrollAreaWidgetContents)
        self.error_graph.setMinimumSize(QtCore.QSize(0, 400))
        self.error_graph.setObjectName("error_graph")
        self.gridLayout_4.addWidget(self.error_graph, 3, 0, 1, 1)
        self.error_graph_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.error_graph_label.setObjectName("error_graph_label")
        self.gridLayout_4.addWidget(self.error_graph_label, 2, 0, 1, 1)
        self.var_subset_graph_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.var_subset_graph_label.setObjectName("var_subset_graph_label")
        self.gridLayout_4.addWidget(self.var_subset_graph_label, 0, 0, 1, 1)
        self.exp_graph_scroll.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_3.addWidget(self.exp_graph_scroll, 0, 0, 1, 4)
        self.tab_panel.addTab(self.experimenttab, "")
        self.horizontalLayout.addWidget(self.tab_panel)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 952, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.distributions_label.setBuddy(self.distributions_list)
        self.true_model_label.setBuddy(self.true_model_panel)
        self.num_samples_max_label.setBuddy(self.num_samples_max_input)
        self.num_samples_min_label.setBuddy(self.num_samples_min_input)
        self.num_samples_step_label.setBuddy(self.num_samples_step_input)
        self.num_trials_label.setBuddy(self.num_trials_input)
        self.error_graph_label.setBuddy(self.error_graph)
        self.var_subset_graph_label.setBuddy(self.variable_subset_graph)

        self.retranslateUi(MainWindow)
        self.tab_panel.setCurrentIndex(0)
        self.add_var_button.clicked.connect(self.distributions_list.add_variable)
        self.fit_distributions_button.clicked.connect(self.distributions_list.fit_distributions)
        self.fit_distributions_button.clicked.connect(self.correlations_table.fit_distributions)
        self.upload_button.clicked.connect(self.distributions_list.upload_file)
        self.distributions_list.rowsChanged.connect(self.correlations_table.update_variables)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simulation"))
        self.distributions_label.setText(_translate("MainWindow", "Distributions"))
        self.correlations_label.setText(_translate("MainWindow", "Correlations"))
        self.upload_button.setText(_translate("MainWindow", "Upload Data"))
        self.true_model_panel.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">x1 + x2</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.add_var_button.setText(_translate("MainWindow", "Add Variable"))
        self.fit_distributions_button.setText(_translate("MainWindow", "Fit Distributions"))
        self.true_model_label.setText(_translate("MainWindow", "True Model"))
        self.groupBox_2.setTitle(_translate("MainWindow", "True Model Fitting"))
        self.lasso_cv_true_model.setText(_translate("MainWindow", "Lasso CV"))
        self.fit_true_model_button.setText(_translate("MainWindow", "Fit True Model"))
        self.tab_panel.setTabText(self.tab_panel.indexOf(self.datatab), _translate("MainWindow", "Data"))
        self.num_samples_max_label.setText(_translate("MainWindow", "# Samples Max"))
        self.num_samples_max_input.setText(_translate("MainWindow", "50"))
        self.num_trials_input.setText(_translate("MainWindow", "50"))
        self.num_samples_min_input.setText(_translate("MainWindow", "20"))
        self.num_samples_min_label.setText(_translate("MainWindow", "# Samples Min"))
        self.num_samples_step_label.setText(_translate("MainWindow", "# Samples Step"))
        self.num_samples_step_input.setText(_translate("MainWindow", "5"))
        self.num_trials_label.setText(_translate("MainWindow", "Number of Trials:"))
        self.run_experiment_button.setText(_translate("MainWindow", "Run Experiment"))
        self.groupBox.setTitle(_translate("MainWindow", "Error Metrics"))
        self.prediction_mse_checkbox.setText(_translate("MainWindow", "Prediction MSE"))
        self.matthews_coef_checkbox.setText(_translate("MainWindow", "Matthew\'s Coefficient"))
        self.selection_methods_group.setTitle(_translate("MainWindow", "Selection Methods"))
        self.lasso_cv_checkbox.setText(_translate("MainWindow", "LassoCV"))
        self.lasso_cv_std_checkbox.setText(_translate("MainWindow", "Lasso CV + 1 Std. Dev"))
        self.lasso_bic_checkbox.setText(_translate("MainWindow", "Lasso BIC"))
        self.graphics_metrics_group.setTitle(_translate("MainWindow", "Subset Metrics"))
        self.perfectly_chosen_checkbox.setText(_translate("MainWindow", "% Perfectly Chosen"))
        self.predictors_missed_checkbox.setText(_translate("MainWindow", "# Predictors Missed"))
        self.false_predictors_checkbox.setText(_translate("MainWindow", "# False Predictors"))
        self.symm_diff_checkbox.setText(_translate("MainWindow", "Mean Symmetric Diff."))
        self.error_graph_label.setText(_translate("MainWindow", "Error Graph"))
        self.var_subset_graph_label.setText(_translate("MainWindow", "Variable Subset Graph"))
        self.tab_panel.setTabText(self.tab_panel.indexOf(self.experimenttab), _translate("MainWindow", "Experiment"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

from components import CorrelationsTable, DistributionsList
