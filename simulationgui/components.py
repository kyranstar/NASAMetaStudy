from PyQt5.QtWidgets import QMainWindow, QListWidget, QTableView, QListWidgetItem, QWidget, QTableWidgetItem
from PyQt5.QtCore import pyqtSignal, QAbstractTableModel, Qt, QVariant, QModelIndex
from dist_list_widget import Ui_DistListItem
from gen_ui import Ui_MainWindow
import pandas as pd


class DistListItem(QWidget, Ui_DistListItem):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.setupUi(self)


class DistributionsList(QListWidget):

    rowAdded = pyqtSignal(list)

    def __init__(self, parent):
        super().__init__(parent)

    def get_names(self):
        """
        Returns a list of the names of the variables stored in the distributions list.
        """
        num_items = self.model().rowCount()
        return [self.itemWidget(self.item(i)).name_input.text() for i in range(num_items)]

    def add_variable(self):
        print("Adding var")
        item = QListWidgetItem(self)
        child_item = DistListItem()

        # Whenever a variable name is edited, we update the correlation table
        child_item.name_input.editingFinished.connect(
            lambda: self._get_super_parent().correlations_table.update_variables(self.get_names()))

        item.setSizeHint(child_item.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, child_item)
        self.rowAdded.emit(self.get_names())

    def _get_super_parent(self):
        curr_ob = self.parentWidget()
        while not isinstance(curr_ob, Ui_MainWindow) and not curr_ob is None:
            curr_ob = curr_ob.parentWidget()
        return curr_ob


class EditablePandasModel(QAbstractTableModel):
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
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def set_dataframe(self, df):
        self.layoutAboutToBeChanged.emit()
        self._df = df

        self.layoutChanged.emit()

    def sort_by(self, names):
        self.layoutAboutToBeChanged.emit()
        sorted_cols = sorted(self._df.index, key=names.index)
        self._df = self._df[sorted_cols].reindex(sorted_cols)
        self.layoutChanged.emit()

    def flags(self, index):
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable


class CorrelationsTable(QTableView):
    def __init__(self, parent):
        super().__init__(parent)
        self.setModel(EditablePandasModel())

    def update_variables(self, names):
        new_df = self.model()._df
        existing_names = new_df.columns
        # Add rows and columns if they don't exist already
        for col in [name for name in names if not name in existing_names]:
            print("Adding %s" % col)
            new_df[col] = 0.0
            #new_df = new_df.set_index(col, append=True)
            row = {}
            for col in new_df.columns:
                row[col] = 0.0
            #row["Name"] = name
            new_df.loc[col] = row

        # delete rows and columns we shouldn't have
        cols_to_delete = [name for name in existing_names if not name in names]
        new_df = new_df.drop(cols_to_delete, axis=0).drop(cols_to_delete, axis=1)

        self.model().set_dataframe(new_df)
        self.model().sort_by(names)
