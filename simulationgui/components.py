from PyQt5.QtWidgets import QMainWindow, QListWidget, QTableView, QListWidgetItem, QWidget
from PyQt5.QtCore import pyqtSignal
from dist_list_widget import Ui_DistListItem
from gen_ui import Ui_MainWindow


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


class CorrelationsTable(QTableView):
    def __init__(self, parent):
        super().__init__(parent)

    def update_variables(self, names):
        print("Updating corr table: " + str(names))
