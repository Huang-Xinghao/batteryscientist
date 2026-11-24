"""Alias validation must preserve ordinary mechanism descriptions."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import json

from batteryscientist.tools.wiki.relations import validate_knowledge_prose
from batteryscientist.tools.wiki.ingest import WikiIngestion


class WikiValidation(unittest.TestCase):
    def test_common_words_are_not_source_aliases(self):
        terms = [{'name': 'Add', 'description': 'Separate additive branches.'},
                 {'name': 'Patch', 'description': 'Temporal patch representations.'}]
        validate_knowledge_prose(
            {'experiments': [{'question': 'Does conditioning add practical value?'}]},
            {'Mechanism': 'A patch representation of the history.'}, terms)

    def test_exact_alias_still_rejected_in_knowledge_prose(self):
        for name in ['Add', 'Patch', 'SmallMLP', '自定义模块']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_knowledge_prose({}, {'Mechanism': f'Use {name} for prediction.'},
                                        [{'name': name, 'description': 'A mechanism.'}])

    def test_alias_metadata_and_original_abstract_remain_allowed(self):
        terms = [{'name': 'Patch', 'description': 'Temporal blocks.'}]
        validate_knowledge_prose({'coined_terms': terms},
                                {'Abstract': 'We propose Patch.', 'Mechanism': 'Temporal blocks.'}, terms)
        validate_knowledge_prose({}, {'Mechanism': 'PatchTST and additive heads.'}, terms)

    def test_validation_failure_returns_existing_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = root / 'draft.json'
            draft.write_text('{}')
            wiki = WikiIngestion.__new__(WikiIngestion)
            wiki.project = root
            wiki.job = root
            wiki.receipt = None
            wiki.review_count = 0
            wiki._require_prepared = Mock()
            wiki._check_inputs = Mock()
            wiki._next_offset = Mock(return_value=None)
            wiki._load_draft = Mock(return_value={})
            wiki._save_state = Mock()
            wiki._validate = Mock(side_effect=ValueError('Invalid draft'))
            wiki._display = lambda p: str(p.relative_to(root))
            wiki._result = lambda value, error: value
            result = wiki.wiki_review(None, 'draft.json')
            report = root / result['report_path']
            self.assertTrue(report.is_file())
            self.assertEqual(json.loads(report.read_text()), result['report'])
            self.assertIsNone(wiki.approval)


if __name__ == '__main__':
    unittest.main()
