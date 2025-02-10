# agent/gui_success.py
"""
GUI for managing the 'successful_exchanges' table in the local SQLite DB.
This implementation uses PySide6 to provide a modern, feature-rich interface.
It allows the user to search (across both user prompts and assistant responses),
edit (with a dedicated dialog preserving formatting), add, update, remove, and export records.
"""

import sys
import os
import logging
import threading
import pandas as pd
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLineEdit, QLabel,
    QFileDialog, QMessageBox, QHeaderView, QInputDialog, QDialog, QTextEdit, QDialogButtonBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from agent.db import ConversationDB

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EditResponseDialog(QDialog):
    def __init__(self, exchange_id, user_prompt, current_response, parent=None):
        super().__init__(parent)
        self.exchange_id = exchange_id
        self.user_prompt = user_prompt
        self.setWindowTitle(f"Edit Response for Exchange ID {exchange_id}")
        self.setMinimumSize(600, 400)
        self.init_ui(current_response)

    def init_ui(self, current_response):
        layout = QVBoxLayout(self)
        
        prompt_label = QLabel("User Prompt:")
        layout.addWidget(prompt_label)
        prompt_edit = QTextEdit()
        prompt_edit.setPlainText(self.user_prompt)
        prompt_edit.setReadOnly(True)
        prompt_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(prompt_edit)
        
        response_label = QLabel("Assistant Response:")
        layout.addWidget(response_label)
        self.response_edit = QTextEdit()
        self.response_edit.setPlainText(current_response)
        # Use a monospaced font to preserve code formatting.
        self.response_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(self.response_edit)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_new_response(self):
        return self.response_edit.toPlainText().strip()

class SuccessDBEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Successful Exchanges Manager")
        self.resize(1000, 600)
        self.db = ConversationDB()
        self.init_ui()

    def init_ui(self):
        # Main widget and layout.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Search bar layout.
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setFont(QFont("Arial", 12))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in User Prompt and Response...")
        self.search_input.setFont(QFont("Arial", 12))
        self.search_input.textChanged.connect(self.refresh_table)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.search_input.clear())
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(clear_btn)
        main_layout.addLayout(search_layout)
        
        # Table widget for displaying exchanges.
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "User Prompt", "Response", "Timestamp"])
        # Enable interactive resizing.
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        # Set equal widths for User Prompt and Response columns
        self.table.setColumnWidth(1, 400)  # User Prompt column width
        self.table.setColumnWidth(2, 400)  # Response column width
        
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        # Double-click on a row will open an editor dialog for the assistant response.
        self.table.itemDoubleClicked.connect(self.edit_response_dialog)
        main_layout.addWidget(self.table)
        
        # Buttons layout.
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_table)
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_exchange)
        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self.update_exchange)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_exchange)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.exit_btn)
        main_layout.addLayout(btn_layout)
        
        self.refresh_table()

    def refresh_table(self):
        search_term = self.search_input.text().strip().lower()
        try:
            exchanges = self.db.list_successful_exchanges(search_term) if search_term else self.db.list_successful_exchanges()
        except Exception as e:
            logger.error(f"Error refreshing table: {e}")
            exchanges = []
        self.table.setRowCount(0)
        for ex in exchanges:
            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(ex["id"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(ex["user_prompt"]))
            
            # Prepare the response for display
            response_lines = ex["assistant_response"].splitlines()
            if len(response_lines) > 1:
                display_response = "\n".join(response_lines[1:])  # Skip the first line for display
            else:
                display_response = ex["assistant_response"]  # If there's only one line, show it as is
            
            self.table.setItem(row_idx, 2, QTableWidgetItem(display_response))
            self.table.setItem(row_idx, 3, QTableWidgetItem(ex["timestamp"]))

            # Store the original response in the item for later retrieval
            self.table.item(row_idx, 2).setData(Qt.UserRole, ex["assistant_response"])

    def add_exchange(self):
        prompt, ok1 = QInputDialog.getMultiLineText(self, "Add Exchange", "Enter user prompt:")
        if not ok1 or not prompt.strip():
            return
        response, ok2 = QInputDialog.getMultiLineText(self, "Add Exchange", "Enter assistant response:")
        if not ok2 or not response.strip():
            return
        try:
            self.db.add_successful_exchange(prompt.strip(), response.strip())
            QMessageBox.information(self, "Success", "Exchange added successfully.")
            self.refresh_table()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add exchange: {e}")

    def update_exchange(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Update Exchange", "Please select an exchange to update.")
            return
        exchange_id = int(self.table.item(self.table.currentRow(), 0).text())
        current_response = self.table.item(self.table.currentRow(), 2).text()
        new_response, ok = QInputDialog.getMultiLineText(self, "Update Exchange", "Enter new assistant response:", current_response)
        if ok and new_response.strip():
            try:
                self.db.update_successful_exchange(exchange_id, new_response.strip())
                QMessageBox.information(self, "Update Exchange", f"Exchange ID {exchange_id} updated.")
                self.refresh_table()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update exchange: {e}")

    def remove_exchange(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Remove Exchange", "Please select an exchange to remove.")
            return
        exchange_id = int(self.table.item(self.table.currentRow(), 0).text())
        reply = QMessageBox.question(self, "Remove Exchange", f"Are you sure you want to remove exchange ID {exchange_id}?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self.db.remove_successful_exchange(exchange_id)
                QMessageBox.information(self, "Remove Exchange", f"Exchange ID {exchange_id} removed.")
                self.refresh_table()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove exchange: {e}")

    def export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export to CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_sql_query("SELECT * FROM successful_exchanges ORDER BY timestamp DESC", self.db.conn)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export CSV", f"Data exported successfully to {file_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Export CSV", f"Failed to export data: {e}")

    def edit_response_dialog(self, item):
        # Open a dialog for editing the assistant response of the selected row.
        row = item.row()
        exchange_id = int(self.table.item(row, 0).text())
        user_prompt = self.table.item(row, 1).text()
        
        # Retrieve the original response from the item data
        current_response = self.table.item(row, 2).data(Qt.UserRole)
        
        dlg = EditResponseDialog(exchange_id, user_prompt, current_response, self)
        if dlg.exec() == QDialog.Accepted:
            new_response = dlg.get_new_response()
            try:
                self.db.update_successful_exchange(exchange_id, new_response)
                QMessageBox.information(self, "Update", f"Exchange ID {exchange_id} updated.")
                self.refresh_table()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update exchange: {e}")

    def closeEvent(self, event):
        self.db.close()
        event.accept()

class EditResponseDialog(QDialog):
    def __init__(self, exchange_id, user_prompt, current_response, parent=None):
        super().__init__(parent)
        self.exchange_id = exchange_id
        self.user_prompt = user_prompt
        self.setWindowTitle(f"Edit Response for Exchange ID {exchange_id}")
        self.resize(700, 500)
        self.init_ui(current_response)

    def init_ui(self, current_response):
        layout = QVBoxLayout(self)
        
        prompt_label = QLabel("User Prompt:")
        prompt_label.setFont(QFont("Arial", 12))
        layout.addWidget(prompt_label)
        
        prompt_view = QTextEdit()
        prompt_view.setPlainText(self.user_prompt)
        prompt_view.setReadOnly(True)
        prompt_view.setFont(QFont("Courier New", 10))
        layout.addWidget(prompt_view)
        
        response_label = QLabel("Assistant Response (Editable):")
        response_label.setFont(QFont("Arial", 12))
        layout.addWidget(response_label)
        
        self.response_edit = QTextEdit()
        self.response_edit.setPlainText(current_response)
        self.response_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(self.response_edit)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_new_response(self):
        return self.response_edit.toPlainText().strip()

def launch_success_gui():
    """
    Launch the Success DB Manager GUI.
    This function should be run as a standalone process.
    """
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = SuccessDBEditor()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_success_gui()