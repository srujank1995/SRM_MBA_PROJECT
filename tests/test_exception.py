import sys
import pytest
from unittest.mock import MagicMock
from src.exception import CustomException, error_message_detail


def _make_exc_info(lineno=42, filename="test_file.py"):
    """Build a fake exc_info tuple whose traceback points to a known location."""
    tb = MagicMock()
    tb.tb_lineno = lineno
    tb.tb_frame.f_code.co_filename = filename
    error_detail = MagicMock()
    error_detail.exc_info.return_value = (None, None, tb)
    return error_detail


class TestErrorMessageDetail:
    def test_contains_filename(self):
        detail = _make_exc_info(filename="mymodule.py")
        msg = error_message_detail("some error", detail)
        assert "mymodule.py" in msg

    def test_contains_line_number(self):
        detail = _make_exc_info(lineno=99)
        msg = error_message_detail("boom", detail)
        assert "99" in msg

    def test_contains_error_text(self):
        detail = _make_exc_info()
        msg = error_message_detail("divide by zero", detail)
        assert "divide by zero" in msg

    def test_format_structure(self):
        detail = _make_exc_info(lineno=7, filename="pipeline.py")
        msg = error_message_detail("oops", detail)
        assert "pipeline.py" in msg
        assert "7" in msg
        assert "oops" in msg


class TestCustomException:
    def test_inherits_from_exception(self):
        assert issubclass(CustomException, Exception)

    def test_str_returns_formatted_message(self):
        detail = _make_exc_info(lineno=10, filename="utils.py")
        exc = CustomException("test error", detail)
        result = str(exc)
        assert "utils.py" in result
        assert "10" in result
        assert "test error" in result

    def test_can_be_raised_and_caught(self):
        detail = _make_exc_info()
        with pytest.raises(CustomException):
            raise CustomException("raised error", detail)

    def test_caught_as_base_exception(self):
        detail = _make_exc_info()
        with pytest.raises(Exception):
            raise CustomException("base exception test", detail)

    def test_error_message_attribute_set(self):
        detail = _make_exc_info(lineno=5, filename="ingestion.py")
        exc = CustomException("attr test", detail)
        assert hasattr(exc, "error_message")
        assert "ingestion.py" in exc.error_message
        assert "5" in exc.error_message
