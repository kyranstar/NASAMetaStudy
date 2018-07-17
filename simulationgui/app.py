import sys
import analysis
import datasampling
from gen_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import pyqtSignal
from sklearn.linear_model import Lasso, LinearRegression
import numpy as np
from utility import error
import traceback


class MyMainWindow(QMainWindow, Ui_MainWindow):

    analysisDataUpdated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.subset_metrics = []
        self.subset_methods = []
        self.error_types = []
        self.sample_range = []
        self.run_experiment_button.clicked.connect(self.run_experiment)
        self.fit_true_model_button.clicked.connect(self.fit_true_model)
        self.analysisDataUpdated.connect(self.variable_subset_graph.plot_var_subset)
        self.analysisDataUpdated.connect(self.error_graph.plot_error)

    def fit_true_model(self):
        if self.distributions_list.data_file is None:
            error('No data file specified')
            return

        var_names = self.distributions_list.get_dummy_names()
        dependent_var = self.dependent_var_input.text()
        df = self.distributions_list.data_file_dummy[var_names]
        if not dependent_var in var_names:
            error('Dependent variable "%s" is not a variable in the variable list' %
                  dependent_var)
            return
        # Get selected fitting method
        selected_button = self.true_model_fitting_buttons.checkedButton()
        if selected_button is None:
            error('Select a fitting method')
            return
        fit_intercept = self.true_model_intercept_checkbox.isChecked()
        selected_text = selected_button.text()
        if selected_text == 'Lasso':
            alpha = self.true_model_lasso_parameter.value()
            true_model = Lasso(alpha=alpha, fit_intercept=fit_intercept)
            true_model.fit(df.drop([dependent_var], axis=1), df[[dependent_var]])
            coef = list(np.array(true_model.coef_).flat)
        else:
            error('Not implemented yet')
            return

        forced_vars = [i for i, var_name in enumerate(
            var_names) if var_name in self.distributions_list.get_forced_variables()]
        # Create model string
        true_vars = [i for i, x in enumerate(coef) if x != 0.0] + forced_vars
        unique_true_vars = []
        [unique_true_vars.append(item) for item in true_vars if item not in unique_true_vars]
        # Run ridge regression on included vars
        regress = LinearRegression(fit_intercept=fit_intercept)
        regress.fit(df.drop([dependent_var], axis=1).iloc[:, unique_true_vars], df[[dependent_var]])
        new_coef = list(np.array(regress.coef_).flat)

        predictors = df.drop([dependent_var], axis=1).columns
        model_str = ""
        for picked_var in unique_true_vars:
            model_str += "%f * %s + " % (new_coef[unique_true_vars.index(picked_var)],
                                         predictors[picked_var])
        if fit_intercept:
            model_str += str(true_model.intercept_[0])
        else:
            # Remove trailing +
            model_str = model_str[0:-3]

        # Calculate gaussian error term and add to formula
        calculated_dependent = datasampling.true_model(
            df.drop([dependent_var], axis=1), dependent_var, model_str)[dependent_var]
        error_term = df.loc[:, dependent_var] - calculated_dependent
        stddev = error_term.std()
        model_str += " + normal(0.0, %f)" % (stddev)

        model_str = model_str.replace("+ -", "- ")
        self.true_model_panel.setText(model_str)

    def run_experiment(self):
        try:
            # Read data from GUI
            trials = int(self.num_trials_input.text())
            num_samples_min = int(self.num_samples_min_input.text())
            num_samples_step = int(self.num_samples_step_input.text())
            num_samples_max = int(self.num_samples_max_input.text())
            sample_range = range(num_samples_min, num_samples_max, num_samples_step)

            # TODO clean this up
            subset_methods = []
            if self.lasso_cv_checkbox.isChecked():
                subset_methods.append(analysis.LassoCVMSE())
            if self.lasso_cv_std_checkbox.isChecked():
                subset_methods.append(analysis.LassoCVMSEStd())
            if self.lasso_bic_checkbox.isChecked():
                subset_methods.append(analysis.LassoBIC())
            if self.fitting_lasso_checkbox.isChecked():
                subset_methods.append(analysis.LassoAlpha(self.true_model_lasso_parameter.value()))
            subset_metrics = []
            if self.perfectly_chosen_checkbox.isChecked():
                subset_metrics.append('perfectly_chosen')
            if self.predictors_missed_checkbox.isChecked():
                subset_metrics.append('predictors_missed')
            if self.symm_diff_checkbox.isChecked():
                subset_metrics.append('symm_diff')
            if self.false_predictors_checkbox.isChecked():
                subset_metrics.append('false_predictors_chosen')
            predict_methods = []
            if self.rand_forest_checkbox.isChecked():
                predict_methods.append(analysis.RandomForest())
            error_types = []
            if self.prediction_mse_checkbox.isChecked():
                error_types.append('prediction_mse')
            if self.matthews_coef_checkbox.isChecked():
                print("Not implemented yet")
            true_model_text = self.true_model_panel.toPlainText()
            variables = self.distributions_list.get_names()

            self.sample_range = sample_range
            self.subset_metrics = subset_metrics
            self.subset_methods = subset_methods
            self.predict_methods = predict_methods
            self.error_types = error_types

            forced_var_ind = [i for i, e in enumerate(
                self.distributions_list.get_names()) if e in self.distributions_list.get_forced_variables()]

            data_model = datasampling.DataModel(
                self.distributions_list.get_means(), self.correlations_table.get_table(),
                self.distributions_list.get_names(), self.distributions_list.categorical_portions,
                self.distributions_list.dummy_cols, self.dependent_var_input.text(),
                forced_var_ind)
            self.analysis_data = analysis.subset_accuracy(
                variables, data_model, true_model_text, sample_range, trials, subset_metrics, subset_methods, predict_methods, error_types)
            # TODO update graph as data collected
            self.analysisDataUpdated.emit()
        except Exception as e:
            error(str(e))
            traceback.print_exc()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
