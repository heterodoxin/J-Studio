import unittest
from pathlib import Path

HTML = Path(__file__).with_name("j-studio.html")
DOC = Path(__file__).parents[1] / "superpowers/specs/2026-07-07-j-studio-ui-design.md"


class CheatEngineFidelityMockupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = HTML.read_text(encoding="utf-8")
        cls.doc = DOC.read_text(encoding="utf-8")

    def test_uses_cheat_engine_main_window_geometry(self):
        self.assertIn('class="ce-window"', self.html)
        self.assertIn("width:734px", self.html)
        self.assertIn("height:592px", self.html)

    def test_has_compact_native_menu_and_model_toolbar(self):
        for name in ("File", "Edit", "Model", "Table", "Help"):
            self.assertIn(f">{name}<", self.html)
        self.assertIn('id="modelLabel"', self.html)
        self.assertIn("Qwen3.6-27B", self.html)

    def test_upper_pane_matches_found_list_and_scan_controls(self):
        self.assertIn('id="foundList"', self.html)
        for heading in ("Term", "Score", "Previous"):
            self.assertIn(f">{heading}<", self.html)
        self.assertIn("J-Space Scan Options", self.html)
        for label in ("First Read", "Next Read", "Undo Read"):
            self.assertIn(label, self.html)

    def test_lower_pane_is_intervention_list(self):
        self.assertIn('id="interventionList"', self.html)
        self.assertIn("Add Intervention Manually", self.html)
        self.assertIn("Model View", self.html)

    def test_generation_reuses_scan_buttons_for_live_control(self):
        self.assertIn("Pause", self.html)
        self.assertIn("Next Token", self.html)
        self.assertIn("Stop", self.html)
        self.assertIn("setRunMode", self.html)

    def test_document_tabs_include_main_chat_jlens_and_rules(self):
        for tab in ("Main", "Chat", "J-Lens", "Rules"):
            self.assertIn(f'data-tab="{tab.lower()}"', self.html)

    def test_chat_tab_can_send_prompts_and_show_outputs(self):
        self.assertIn('id="chatInput"', self.html)
        self.assertIn('id="chatMessages"', self.html)
        self.assertIn("sendChat", self.html)

    def test_jlens_tab_matches_reference_visualization_structure(self):
        for region in (
            "jlensMatrix",
            "byLayer",
            "byPosition",
            "rankHeatmap",
            "layerPlot",
            "positionPlot",
        ):
            self.assertIn(f'id="{region}"', self.html)

    def test_intervention_editor_supports_precise_inject_and_replace(self):
        self.assertIn('id="interventionEditor"', self.html)
        self.assertIn("Match term", self.html)
        self.assertIn("Replacement term", self.html)
        self.assertIn("Layer range", self.html)
        self.assertIn("Apply timing", self.html)

    def test_design_doc_requires_cheat_engine_fidelity(self):
        self.assertIn("## 4. Cheat Engine Fidelity Contract", self.doc)
        self.assertIn("734 x 572", self.doc)
        self.assertIn("First Read", self.doc)
        self.assertIn("double-clicking a found concept", self.doc.lower())

    def test_design_doc_links_to_html_reference_mockup(self):
        self.assertIn("../../mockups/j-studio.html", self.doc)
        self.assertIn("Chat tab", self.doc)
        self.assertIn("J-Lens tab", self.doc)


if __name__ == "__main__":
    unittest.main()
