from PySide6.QtCore import Qt

from jstudio.domain import RuleTrigger
from jstudio.ui.rules.workspace import RulesWorkspace

VALID_RULE = """function run(ctx) {
  if (ctx.jspace.has("injection", {minScore: 0.7})) {
    return [jspace.replace("injection", "trusted", {
      strength: 0.8, layers: "current", duration: "next-token", matchMode: "exact"
    })];
  }
  return [];
}"""


def test_rules_workspace_has_required_panes(qtbot, services, project):
    workspace = RulesWorkspace(services.rules, project)
    qtbot.addWidget(workspace)

    assert workspace.main_splitter.count() == 3
    assert [workspace.side_tabs.tabText(i) for i in range(2)] == [
        "API",
        "Test",
    ]
    assert not any(
        workspace.side_tabs.tabText(i) == "Configuration"
        for i in range(workspace.side_tabs.count())
    )
    assert [workspace.output_tabs.tabText(i) for i in range(3)] == [
        "Problems",
        "Returned Actions",
        "Execution Log",
    ]


def test_rules_api_tab_documents_context_helpers_and_examples(qtbot, services, project):
    workspace = RulesWorkspace(services.rules, project)
    qtbot.addWidget(workspace)

    text = workspace.api_reference.toPlainText()

    assert "Define constants in your rule source" in text
    assert "ctx.generation.outputText" in text
    assert "jspace.inject" in text
    assert "generation.stop" in text
    assert "function run(ctx)" in text
    assert "ctx.config" not in text


def test_edited_rule_requires_current_successful_test_before_enable(
    qtbot, services, project
):
    workspace = RulesWorkspace(services.rules, project)
    qtbot.addWidget(workspace)
    workspace.new_rule("Guard")
    workspace.editor.setPlainText(VALID_RULE)
    assert not workspace.enable_action.isEnabled()

    qtbot.mouseClick(workspace.test_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.last_test_passed, timeout=3000)

    assert workspace.enable_action.isEnabled()
    workspace.editor.insertPlainText("\n// changed")
    assert not workspace.enable_action.isEnabled()


def test_three_failures_disable_rule(qtbot, services, project):
    workspace = RulesWorkspace(services.rules, project)
    qtbot.addWidget(workspace)
    rule = workspace.add_rule(
        name="Guard",
        source=VALID_RULE,
        trigger=RuleTrigger.JSPACE_FRAME,
        enabled=True,
    )

    for _ in range(3):
        workspace.record_execution_failure(rule.rule_id, "timeout")

    assert not workspace.rule(rule.rule_id).enabled
    assert workspace.rule(rule.rule_id).consecutive_failures == 3


def test_rule_checkbox_updates_project_state(qtbot, services, project):
    workspace = RulesWorkspace(services.rules, project)
    qtbot.addWidget(workspace)
    workspace.add_rule(
        name="Guard",
        source=VALID_RULE,
        trigger=RuleTrigger.BEFORE_TOKEN,
        enabled=False,
    )

    accepted = workspace.rule_model.setData(
        workspace.rule_model.index(0, 0),
        Qt.CheckState.Checked,
        Qt.ItemDataRole.CheckStateRole,
    )

    assert accepted
    assert project.rules[0].enabled


def test_successful_test_shows_validated_actions_and_metrics(qtbot, services, project):
    workspace = RulesWorkspace(services.rules, project)
    qtbot.addWidget(workspace)
    workspace.new_rule("Guard")
    workspace.editor.setPlainText(VALID_RULE)

    qtbot.mouseClick(workspace.test_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.last_test_passed, timeout=3000)

    assert "replace" in workspace.returned_actions.toPlainText()
    assert "ms" in workspace.execution_log.toPlainText()
