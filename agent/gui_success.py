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
from typing import List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLineEdit, QLabel,
    QFileDialog, QMessageBox, QHeaderView, QInputDialog, QDialog, QTextEdit, QDialogButtonBox,
    QStatusBar, QProgressDialog
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QTimer

from agent.db import ConversationDB

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EditResponseDialog(QDialog):
    def __init__(self, exchange_id, user_prompt, current_response, parent=None):
        super().__init__(parent)
        self.exchange_id = exchange_id
        self.setWindowTitle(f"Edit Exchange ID {exchange_id}")
        self.setMinimumSize(600, 400)
        self.init_ui(user_prompt, current_response)

    def init_ui(self, user_prompt, current_response):
        layout = QVBoxLayout(self)
        
        prompt_label = QLabel("User Prompt:")
        layout.addWidget(prompt_label)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(user_prompt)
        self.prompt_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(self.prompt_edit)
        
        response_label = QLabel("Assistant Response:")
        layout.addWidget(response_label)
        self.response_edit = QTextEdit()
        self.response_edit.setPlainText(current_response)
        self.response_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(self.response_edit)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_new_values(self):
        return {
            'prompt': self.prompt_edit.toPlainText().strip(),
            'response': self.response_edit.toPlainText().strip()
        }

class SuccessDBEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Successful Exchanges Manager")
        self.resize(1000, 600)
        self.db = ConversationDB()
        self.batch_size = 100  # Number of records to load at once
        self.current_offset = 0
        self.total_records = 0
        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Search bar layout with debounce timer
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setFont(QFont("Arial", 12))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in User Prompt and Response...")
        self.search_input.setFont(QFont("Arial", 12))
        
        # Add debounce timer for search
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.refresh_table)
        self.search_input.textChanged.connect(self.on_search_changed)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_search)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(clear_btn)
        main_layout.addLayout(search_layout)
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Table widget with optimized settings
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "User Prompt", "Response", "Timestamp"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnWidth(1, 400)
        self.table.setColumnWidth(2, 400)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)  # Allow multiple selection
        self.table.itemDoubleClicked.connect(self.edit_response_dialog)
        
        # Add scroll event handling for infinite scroll
        self.table.verticalScrollBar().valueChanged.connect(self.handle_scroll)
        main_layout.addWidget(self.table)
        
        # Buttons layout with batch operations
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(lambda: self.refresh_table(force=True))
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_exchange)
        self.batch_update_btn = QPushButton("Batch Update")
        self.batch_update_btn.clicked.connect(self.batch_update_exchanges)
        self.batch_delete_btn = QPushButton("Batch Delete")
        self.batch_delete_btn.clicked.connect(self.batch_delete_exchanges)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.batch_update_btn)
        btn_layout.addWidget(self.batch_delete_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.exit_btn)
        main_layout.addLayout(btn_layout)
        
        # Initialize the table
        self.refresh_table()

    def on_search_changed(self):
        """Debounce search input to prevent excessive database queries"""
        self.search_timer.stop()
        self.search_timer.start(300)  # Wait 300ms before refreshing

    def clear_search(self):
        """Clear search and reset table"""
        self.search_input.clear()
        self.current_offset = 0
        self.refresh_table(force=True)

    def handle_scroll(self, value):
        """Handle infinite scroll"""
        if value >= self.table.verticalScrollBar().maximum() * 0.9:
            self.load_more_data()

    def load_more_data(self):
        """Load next batch of data"""
        self.current_offset += self.batch_size
        self.refresh_table(append=True)

    def refresh_table(self, force=False, append=False):
        """Refresh table with optimized loading"""
        if not append:
            self.current_offset = 0
            self.table.setRowCount(0)
        
        search_term = self.search_input.text().strip().lower()
        try:
            exchanges = self.db.list_successful_exchanges(
                search_term, 
                offset=self.current_offset,
                limit=self.batch_size
            )
            
            start_row = self.table.rowCount()
            for ex in exchanges:
                row_idx = self.table.rowCount()
                self.table.insertRow(row_idx)
                self.table.setItem(row_idx, 0, QTableWidgetItem(str(ex["id"])))
                self.table.setItem(row_idx, 1, QTableWidgetItem(ex["user_prompt"]))
                
                # Optimize response display
                response_text = self.format_response_for_display(ex["assistant_response"])
                response_item = QTableWidgetItem(response_text)
                response_item.setData(Qt.UserRole, ex["assistant_response"])
                self.table.setItem(row_idx, 2, response_item)
                self.table.setItem(row_idx, 3, QTableWidgetItem(ex["timestamp"]))
            
            # Update status bar
            self.update_status_bar()
            
        except Exception as e:
            logger.error(f"Error refreshing table: {e}")
            QMessageBox.critical(self, "Error", f"Failed to refresh table: {e}")

    def format_response_for_display(self, response: str) -> str:
        """Format response text for optimal display"""
        lines = response.splitlines()
        if len(lines) > 1:
            return "\n".join(lines[1:])
        return response

    def update_status_bar(self):
        """Update status bar with record count and other info"""
        try:
            total = self.db.get_total_exchanges_count()
            current = self.table.rowCount()
            self.status_bar.showMessage(
                f"Showing {current} of {total} exchanges | Last updated: {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            logger.error(f"Error updating status bar: {e}")

    def batch_update_exchanges(self):
        """Handle batch update of selected exchanges"""
        selected_rows = self.get_selected_rows()
        if not selected_rows:
            QMessageBox.warning(self, "Batch Update", "Please select exchanges to update.")
            return
            
        new_response, ok = QInputDialog.getMultiLineText(
            self, "Batch Update", 
            f"Enter new response for {len(selected_rows)} exchanges:"
        )
        if ok and new_response.strip():
            try:
                for row in selected_rows:
                    exchange_id = int(self.table.item(row, 0).text())
                    self.db.update_successful_exchange(exchange_id, new_response.strip())
                
                QMessageBox.information(
                    self, "Batch Update", 
                    f"Successfully updated {len(selected_rows)} exchanges."
                )
                self.refresh_table(force=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update exchanges: {e}")

    def batch_delete_exchanges(self):
        """Handle batch deletion of selected exchanges"""
        selected_rows = self.get_selected_rows()
        if not selected_rows:
            QMessageBox.warning(self, "Batch Delete", "Please select exchanges to delete.")
            return
            
        reply = QMessageBox.question(
            self, "Batch Delete", 
            f"Are you sure you want to delete {len(selected_rows)} exchanges?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Get all exchange IDs to delete
                exchange_ids = [int(self.table.item(row, 0).text()) for row in selected_rows]
                
                # Attempt batch deletion
                success, deleted_ids = self.db.batch_delete_exchanges(exchange_ids)
                
                if success:
                    if len(deleted_ids) == len(exchange_ids):
                        QMessageBox.information(
                            self, "Batch Delete", 
                            f"Successfully deleted all {len(deleted_ids)} exchanges."
                        )
                    else:
                        QMessageBox.warning(
                            self, "Batch Delete", 
                            f"Partially successful: Deleted {len(deleted_ids)} out of {len(exchange_ids)} exchanges."
                        )
                else:
                    QMessageBox.critical(
                        self, "Batch Delete",
                        "Failed to delete any exchanges. Check the logs for details."
                    )
                
                self.refresh_table(force=True)
                
            except Exception as e:
                logger.error(f"Error during batch delete: {e}")
                QMessageBox.critical(self, "Error", f"Failed to delete exchanges: {e}")

    def delete_single_exchange(self, exchange_id: int) -> bool:
        """Delete a single exchange with confirmation"""
        reply = QMessageBox.question(
            self, "Delete Exchange", 
            f"Are you sure you want to delete exchange ID {exchange_id}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if self.db.remove_successful_exchange(exchange_id):
                    QMessageBox.information(
                        self, "Delete Exchange",
                        f"Successfully deleted exchange ID {exchange_id}."
                    )
                    return True
                else:
                    QMessageBox.warning(
                        self, "Delete Exchange",
                        f"Exchange ID {exchange_id} not found or already deleted."
                    )
            except Exception as e:
                logger.error(f"Error deleting exchange {exchange_id}: {e}")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to delete exchange: {e}"
                )
        return False

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Delete:
            # Get selected rows
            selected_rows = self.get_selected_rows()
            if selected_rows:
                if len(selected_rows) == 1:
                    # Single deletion
                    exchange_id = int(self.table.item(selected_rows[0], 0).text())
                    if self.delete_single_exchange(exchange_id):
                        self.refresh_table(force=True)
                else:
                    # Batch deletion
                    self.batch_delete_exchanges()
        else:
            super().keyPressEvent(event)

    def get_selected_rows(self) -> List[int]:
        """Get list of selected row indices"""
        return sorted(set(item.row() for item in self.table.selectedItems()))

    def export_csv(self):
        """Export data to CSV with progress dialog"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export to CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
            
        try:
            progress = QProgressDialog("Exporting data...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            
            def export_worker():
                try:
                    df = pd.read_sql_query(
                        "SELECT * FROM successful_exchanges ORDER BY timestamp DESC",
                        self.db.conn
                    )
                    df.to_csv(file_path, index=False)
                    return True
                except Exception as e:
                    logger.error(f"Export error: {e}")
                    return False
            
            thread = threading.Thread(target=export_worker)
            thread.start()
            
            while thread.is_alive():
                QApplication.processEvents()
                if progress.wasCanceled():
                    return
            
            if thread.is_alive():
                QMessageBox.information(
                    self, "Export CSV",
                    f"Data exported successfully to {file_path}"
                )
            else:
                QMessageBox.critical(
                    self, "Export CSV",
                    "Failed to export data. Check the logs for details."
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Export CSV", f"Failed to export data: {e}")

    def closeEvent(self, event):
        """Clean up resources before closing"""
        try:
            self.search_timer.stop()
            self.db.close()
            event.accept()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            event.accept()

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

    def edit_response_dialog(self, item):
        # Open a dialog for editing both the prompt and response of the selected row.
        row = item.row()
        exchange_id = int(self.table.item(row, 0).text())
        user_prompt = self.table.item(row, 1).text()
        current_response = self.table.item(row, 2).data(Qt.UserRole)
        
        dlg = EditResponseDialog(exchange_id, user_prompt, current_response, self)
        if dlg.exec() == QDialog.Accepted:
            new_values = dlg.get_new_values()
            try:
                self.db.update_successful_exchange(
                    exchange_id, 
                    new_response=new_values['response'],
                    new_prompt=new_values['prompt']
                )
                QMessageBox.information(self, "Update", f"Exchange ID {exchange_id} updated.")
                self.refresh_table()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update exchange: {e}")

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