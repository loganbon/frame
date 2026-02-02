"""Tests for DataFrame validation."""

import pandas as pd
import pytest

from frame.validation import ValidationError, validate_dataframe


class TestValidateDataframe:
    """Test validate_dataframe function."""

    def test_valid_pandas_multiindex(self):
        """Valid DataFrame with MultiIndex passes."""
        df = pd.DataFrame({
            "as_of_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "id": [0, 1],
            "value": [1.0, 2.0],
        }).set_index(["as_of_date", "id"])
        validate_dataframe(df)  # Should not raise

    def test_valid_pandas_columns(self):
        """Valid DataFrame with columns passes."""
        df = pd.DataFrame({
            "as_of_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "id": [0, 1],
            "value": [1.0, 2.0],
        })
        validate_dataframe(df)  # Should not raise

    def test_missing_as_of_date(self):
        """Missing as_of_date raises ValidationError."""
        df = pd.DataFrame({"id": [0, 1], "value": [1.0, 2.0]})
        with pytest.raises(ValidationError, match="missing 'as_of_date'"):
            validate_dataframe(df)

    def test_missing_id(self):
        """Missing id raises ValidationError."""
        df = pd.DataFrame({
            "as_of_date": pd.to_datetime(["2024-01-01"]),
            "value": [1.0],
        })
        with pytest.raises(ValidationError, match="missing 'id'"):
            validate_dataframe(df)

    def test_duplicate_rows(self):
        """Duplicate (as_of_date, id) rows raise ValidationError."""
        df = pd.DataFrame({
            "as_of_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "id": [0, 0],  # Duplicate id for same date
            "value": [1.0, 2.0],
        }).set_index(["as_of_date", "id"])
        with pytest.raises(ValidationError, match="duplicate"):
            validate_dataframe(df)
