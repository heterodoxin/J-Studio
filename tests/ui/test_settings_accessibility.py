from PySide6.QtGui import QPalette

from jstudio.ui.settings import SettingsWindow
from jstudio.ui.theme import apply_dark_palette, apply_light_palette


def test_settings_has_all_categories_and_baseline_default(qtbot):
    settings = SettingsWindow()
    qtbot.addWidget(settings)

    assert [
        settings.categories.item(i).text() for i in range(settings.categories.count())
    ] == [
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
    ]
    assert settings.generation_default.currentText() == "Baseline"
    assert not settings.rule_wall_time.isEnabled()
    assert settings.rule_wall_time.value() == 50


def test_light_and_dark_palettes_keep_readable_roles(qapp):
    apply_light_palette(qapp)
    light = qapp.palette()
    assert light.color(QPalette.ColorRole.Text) != light.color(QPalette.ColorRole.Base)

    apply_dark_palette(qapp)
    dark = qapp.palette()
    assert dark.color(QPalette.ColorRole.Text) != dark.color(QPalette.ColorRole.Base)
    assert dark.color(QPalette.ColorRole.Window).lightness() < 80


def test_icon_only_shell_controls_are_accessibly_named(window):
    for button in (window.session_bar.select_button, window.session_bar.overflow):
        assert button.accessibleName()
        assert button.toolTip()
