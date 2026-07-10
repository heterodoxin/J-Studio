"""Searchable modeless J Studio settings window."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

CATEGORIES = (
    "General",
    "Appearance",
    "Sessions",
    "J-Lens",
    "Interventions",
    "Rules",
    "Generation",
    "Storage",
    "Shortcuts",
    "Languages",
    "Advanced",
)


class SettingsWindow(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("J Studio Settings")
        self.setModal(False)
        self.resize(940, 720)
        root = QVBoxLayout(self)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search settings")
        root.addWidget(self.search)
        body = QHBoxLayout()
        self.categories = QListWidget(self)
        self.categories.addItems(CATEGORIES)
        self.categories.setFixedWidth(190)
        self.pages = QStackedWidget(self)
        self.rule_wall_time = QSpinBox(self)
        self.rule_wall_time.setRange(1, 50)
        self.rule_wall_time.setValue(50)
        self.rule_wall_time.setEnabled(False)
        self.generation_default = QComboBox(self)
        self.generation_default.addItems(["Baseline", "With Armed Stack"])
        self.appearance_mode = QComboBox(self)
        self.appearance_mode.addItems(["System", "Light", "Dark"])
        self.text_scale = QSpinBox(self)
        self.text_scale.setRange(90, 160)
        self.text_scale.setValue(100)
        self.reduced_motion = QComboBox(self)
        self.reduced_motion.addItems(["Follow system", "Reduce motion", "Allow motion"])
        for category in CATEGORIES:
            page = QWidget(self.pages)
            form = QFormLayout(page)
            form.addRow(QLabel(f"<b>{category}</b>"))
            if category == "Appearance":
                form.addRow("Palette", self.appearance_mode)
                form.addRow("Text scale", self.text_scale)
                form.addRow("Motion", self.reduced_motion)
            elif category == "Rules":
                form.addRow("Wall time (ms)", self.rule_wall_time)
                heap = QLineEdit("16 MiB", page)
                heap.setReadOnly(True)
                form.addRow("QuickJS heap", heap)
                form.addRow("Worker", QLabel("Spawned per evaluation"))
                form.addRow(QPushButton("Run Sandbox Self-Test"))
            elif category == "Generation":
                form.addRow("Default run mode", self.generation_default)
            elif category == "Interventions":
                duration = QComboBox(page)
                duration.addItems(["Next Token", "Current Token", "Entire Generation"])
                form.addRow("Default duration", duration)
                form.addRow("Auto-preview", QComboBox(page))
            else:
                form.addRow(QLabel(f"{category} preferences are stored per user."))
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self.pages.addWidget(page)
        body.addWidget(self.categories)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)
        footer = QHBoxLayout()
        footer.addWidget(QPushButton("Restore Page Defaults"))
        footer.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Apply,
            self,
        )
        footer.addWidget(buttons)
        root.addLayout(footer)
        self.categories.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.categories.setCurrentRow(0)
        self.search.textChanged.connect(self._filter)
        buttons.rejected.connect(self.close)

    def _filter(self, text: str) -> None:
        lowered = text.casefold()
        for row in range(self.categories.count()):
            item = self.categories.item(row)
            item.setHidden(bool(lowered and lowered not in item.text().casefold()))
