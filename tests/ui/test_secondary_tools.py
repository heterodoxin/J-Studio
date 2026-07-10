def test_tools_open_focused_secondary_windows(qtbot, window):
    commands = (
        "model_view",
        "layer_explorer",
        "jlens_sweep",
        "influence_trace",
        "generation_trace",
        "experiments",
        "snapshot_manager",
    )
    for command in commands:
        window.commands[command].setEnabled(True)
        window.commands[command].trigger()
        tool = window.tool_window(command)
        assert tool is not None
        assert tool.isVisible()
        assert tool.isWindow()


def test_secondary_tools_have_required_research_surfaces(qtbot, window):
    window.open_tool("influence_trace")
    influence = window.tool_window("influence_trace")
    assert influence.findChild(type(influence.graph), "influenceGraph") is influence.graph
    assert "estimated influence" in influence.disclaimer.text().lower()

    window.open_tool("experiments")
    experiments = window.tool_window("experiments")
    assert [experiments.tabs.tabText(i) for i in range(4)] == [
        "Setup",
        "Runs",
        "Compare",
        "Report",
    ]
