"""Smoke test for the Streamlit demo. Skipped when Streamlit is not installed."""

import importlib.util
import os
import unittest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streamlit_app.py")


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "streamlit not installed")
class StreamlitAppTests(unittest.TestCase):
    def test_app_runs_without_errors(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=180).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.title[0].value, "⚡ Energy demand forecast")
        self.assertGreaterEqual(len(at.metric), 4)


if __name__ == "__main__":
    unittest.main()
