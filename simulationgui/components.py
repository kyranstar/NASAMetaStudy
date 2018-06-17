from PyQt5.QtWidgets import QMainWindow, QListWidget, QTableView, QListWidgetItem, QWidget, QTableWidgetItem, QFileDialog
from PyQt5.QtCore import pyqtSignal, QAbstractTableModel, Qt, QVariant, QModelIndex
from PyQt5.QtGui import QColor, QBrush
from dist_list_widget import Ui_DistListItem
from gen_ui import Ui_MainWindow
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from utility import get_super_parent
import pyqtgraph as pg


class DistListItem(QWidget, Ui_DistListItem):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.setupUi(self)


class DistributionsList(QListWidget):

    # The signal that is emitted whenever a row is added to this list
    rowsChanged = pyqtSignal(list)

    def __init__(self, parent):
        super().__init__(parent)
        self.data_file = None

    def get_names(self):
        """
        Returns a list of the names of the variables stored in the distributions list.
        """
        num_items = self.model().rowCount()
        return [self.itemWidget(self.item(i)).name_input.text() for i in range(num_items)]

    def get_means(self):
        """
        Returns a list of the means of the variables stored in the distributions list.
        """
        num_items = self.model().rowCount()
        return [float(self.itemWidget(self.item(i)).mean_input.text()) for i in range(num_items)]

    def get_variances(self):
        """
        Returns a list of the means of the variables stored in the distributions list.
        """
        num_items = self.model().rowCount()
        return [float(self.itemWidget(self.item(i)).variance_input.text()) for i in range(num_items)]

    def add_variable(self, name=None):
        item = QListWidgetItem(self)
        child_item = DistListItem()
        if name:
            child_item.name_input.setText(name)

        # Whenever a variable name is edited, we update the correlation table
        child_item.name_input.editingFinished.connect(
            lambda: get_super_parent(self).correlations_table.update_variables(self.get_names()))
        # Delete button should delete row
        # TODO fix this, kinda hacky solution
        child_item.delete_button.clicked.connect(
            lambda: self.remove_variable(child_item.delete_button.parent().name_input.text()))

        item.setSizeHint(child_item.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, child_item)
        self.rowsChanged.emit(self.get_names())

    def remove_variable(self, name):
        self.takeItem(self.get_names().index(name))
        self.rowsChanged.emit(self.get_names())

    def fit_distributions(self):
        if self.data_file is None:
            # TODO error message
            print("NO FILE ERROR")
            return
        means = self.data_file.mean(axis=0)
        variance = self.data_file.var(axis=0)
        num_items = self.model().rowCount()
        for i, widget in [(i, self.itemWidget(self.item(i))) for i in range(num_items)]:
            widget.mean_input.setText(str(means[i]))
            widget.variance_input.setText(str(variance[i]))

    def upload_file(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', '.')
        filename = filename[0]
        self.data_file = pd.read_csv(filename)

        current_row_names = self.get_names()

        for new_var in [col for col in self.data_file.columns if not col in current_row_names]:
            self.add_variable(name=new_var)


class CorrelationModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QVariant()
        elif orientation == Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QVariant()

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()

        if not index.isValid():
            return QVariant()

        return QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        self.dataChanged.emit(index, index)
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def set_dataframe(self, df):
        self.layoutAboutToBeChanged.emit()
        self._df = df
        self.dataChanged.emit(self.index(0, 0), self.index(len(df), len(df.columns)))
        self.layoutChanged.emit()

    def sort_by(self, names):
        """
        Sorts the rows and columns of the dataframe by names.
        Arguments:
            names: The list in which order the rows and columns will be sorted.
        """
        self.layoutAboutToBeChanged.emit()
        sorted_cols = sorted(self._df.index, key=names.index)
        self._df = self._df[sorted_cols].reindex(sorted_cols)
        self.layoutChanged.emit()

    def flags(self, index):
        if index.row() == index.column():
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable


class CorrelationsTable(QTableView):
    def __init__(self, parent):
        super().__init__(parent)
        self.setModel(CorrelationModel())
        self.model().dataChanged.connect(self.refresh_heatmap)
        self.colormap = plt.cm.get_cmap('Spectral')

    def update_variables(self, names):
        new_df = self.model()._df
        existing_names = new_df.columns
        # Add rows and columns if they don't exist already
        for name in [name for name in names if not name in existing_names]:
            new_df[name] = 0.0
            #new_df = new_df.set_index(col, append=True)
            row = {}
            for col in new_df.columns:
                row[col] = 0.0
            #row["Name"] = name
            new_df.loc[name] = row
            new_df.loc[name, name] = 1.0

        # delete rows and columns we shouldn't have
        cols_to_delete = [name for name in existing_names if not name in names]
        new_df = new_df.drop(cols_to_delete, axis=0).drop(cols_to_delete, axis=1)

        self.model().set_dataframe(new_df)
        self.model().sort_by(names)

    def fit_distributions(self):
        data_file = get_super_parent(self).distributions_list.data_file
        if data_file is None:
            return
        cov = data_file.corr().loc[self.model()._df.columns, self.model()._df.columns]
        self.model().set_dataframe(cov)

    def get_table(self):
        return self.model()._df

    def refresh_heatmap(self):
        pass
        #print("Refreshing heatmap")
        #norm = matplotlib.colors.Normalize(vmin=self.model()._df.min(), vmax=self.model()._df.max())
        # for i in range(self.model().columnCount()):
        # for j in range(self.model().rowCount()):
        #self.item(i, j).setBackground(QColor(100, 100, 150))
        #self.model().setData(self.model().index(i, j), QBrush(Qt.red), Qt.BackgroundColorRole)


class GraphWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.legend = None

    def plot_var_subset(self):
        parent_widget = get_super_parent(self)
        analysis_data = parent_widget.analysis_data
        subset_methods = parent_widget.subset_methods
        subset_metrics = parent_widget.subset_metrics
        sample_range = parent_widget.sample_range
        self.clear()
        self.setLabel('left', '%')
        self.setLabel('bottom', 'number of samples')
        self.setYRange(0, 100)
        self.setXRange(sample_range[0], sample_range[-1])
        if analysis_data is None:
            return
        # Delete legend if it exists, and create a new one
        if self.legend:
            self.legend.scene().removeItem(self.legend)
        self.legend = self.addLegend()

        i = 0
        for subset_method in subset_methods:
            for subset_metric in subset_metrics:
                key = (subset_method, subset_metric)
                #curve = self.plot()
                # curve.setData(sample_range, analysis_data[key].tolist(
                # ), name="%s: %s" % key, pen=(i, len(subset_methods)*len(subset_metrics)))
                #i += 1
                print(sample_range)
                print(analysis_data[key].tolist())
                item = pg.PlotCurveItem(
                    x=list(sample_range), y=analysis_data[key].values, name="%s: %s" % key, pen=(i, len(subset_methods)*len(subset_metrics)))
                self.addItem(item)
                i += 1
        self.repaint()

    def plot_error(self):
        parent_widget = get_super_parent(self)
        analysis_data = parent_widget.analysis_data
        error_types = parent_widget.error_types
        if analysis_data is None:
            return
