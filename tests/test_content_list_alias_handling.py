import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def install_import_stubs():
    lightrag_module = types.ModuleType("lightrag")
    lightrag_utils = types.ModuleType("lightrag.utils")
    lightrag_utils.logger = FakeLogger()
    lightrag_utils.compute_mdhash_id = lambda value, prefix="": f"{prefix}test"

    raganything_pkg = types.ModuleType("raganything")
    raganything_pkg.__path__ = [str(PROJECT_ROOT / "raganything")]

    raganything_base = types.ModuleType("raganything.base")

    raganything_parser = types.ModuleType("raganything.parser")
    raganything_parser.MineruParser = object
    raganything_parser.MineruExecutionError = RuntimeError
    raganything_parser.get_parser = lambda *args, **kwargs: None

    sys.modules.setdefault("lightrag", lightrag_module)
    sys.modules.setdefault("lightrag.utils", lightrag_utils)
    sys.modules.setdefault("raganything", raganything_pkg)
    sys.modules.setdefault("raganything.base", raganything_base)
    sys.modules.setdefault("raganything.parser", raganything_parser)
    raganything_base.DocStatus = types.SimpleNamespace(
        READY="ready",
        HANDLING="handling",
        PENDING="pending",
        PROCESSING="processing",
        PROCESSED="processed",
        FAILED="failed",
    )


def load_project_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


install_import_stubs()
utils_module = load_project_module(
    "raganything.utils", PROJECT_ROOT / "raganything" / "utils.py"
)
processor_module = load_project_module(
    "raganything.processor", PROJECT_ROOT / "raganything" / "processor.py"
)

ProcessorMixin = processor_module.ProcessorMixin
format_table_body = utils_module.format_table_body
get_equation_text_and_format = utils_module.get_equation_text_and_format
get_table_body = utils_module.get_table_body
normalize_caption_list = utils_module.normalize_caption_list
separate_content = utils_module.separate_content


class ContentListAliasHandlingTests(unittest.TestCase):
    def test_separate_content_preserves_original_content_list_index(self):
        content = [
            {"type": "text", "text": "Before"},
            {"type": "image", "img_path": "/tmp/figure.png"},
            {"type": "text", "text": "After"},
            {"type": "table", "table_body": "| A | B |"},
        ]

        _, multimodal = separate_content(content)

        self.assertEqual([item["_content_list_index"] for item in multimodal], [1, 3])
        self.assertNotIn("_content_list_index", content[1])

    def test_table_alias_and_string_caption_feed_chunk_template(self):
        processor = ProcessorMixin()
        chunk = processor._apply_chunk_template(
            "table",
            {
                "table_data": [["Method", "Score"], ["RAGAnything", "95.2"]],
                "table_caption": "Performance table",
                "table_footnote": "Synthetic example",
            },
            "The table compares scores.",
        )

        self.assertEqual(get_table_body({"table_data": [["A", "B"]]}), [["A", "B"]])
        self.assertEqual(format_table_body([["A", "B"]]), "[['A', 'B']]")
        self.assertEqual(normalize_caption_list("Performance table"), ["Performance table"])
        self.assertIn("['Method', 'Score']", chunk)
        self.assertIn("Performance table", chunk)
        self.assertNotIn("P, e, r, f", chunk)
        self.assertIn("Synthetic example", chunk)

    def test_equation_latex_alias_is_not_dropped_from_chunk_template(self):
        processor = ProcessorMixin()
        item = {
            "latex": "E = mc^2",
            "text": "Mass-energy equivalence",
        }

        equation_text, equation_format = get_equation_text_and_format(item)
        chunk = processor._apply_chunk_template("equation", item, "A physics equation.")

        self.assertEqual(equation_text, "E = mc^2\nDescription: Mass-energy equivalence")
        self.assertEqual(equation_format, "LaTeX")
        self.assertIn("E = mc^2", chunk)
        self.assertIn("Mass-energy equivalence", chunk)
        self.assertIn("Format: LaTeX", chunk)


if __name__ == "__main__":
    unittest.main()
