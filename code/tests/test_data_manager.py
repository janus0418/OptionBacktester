"""
Tests for Multi-Source Data Manager.

Tests the data adapter architecture including:
- BaseDataAdapter interface
- CSVDataAdapter file-based data loading
- DoltDataAdapter wrapper
- DataSourceRegistry with fallback
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from backtester.data.data_manager import (
    BaseDataAdapter,
    CSVDataAdapter,
    DoltDataAdapter,
    DataSourceRegistry,
    DataSourceError,
    DataSourceConnectionError,
    DataNotFoundError,
    AllSourcesFailedError,
)


class TestBaseDataAdapter:
    def test_abstract_methods_required(self):
        with pytest.raises(TypeError):
            BaseDataAdapter()

    def test_concrete_implementation(self):
        class ConcreteAdapter(BaseDataAdapter):
            def connect(self) -> bool:
                return True

            def disconnect(self) -> None:
                pass

            @property
            def is_connected(self) -> bool:
                return True

            @property
            def source_name(self) -> str:
                return "test"

            def get_option_chain(
                self, symbol, date, expiration=None, min_dte=None, max_dte=None
            ):
                return pd.DataFrame()

            def get_underlying_price(self, symbol, date) -> float:
                return 100.0

        adapter = ConcreteAdapter()
        assert adapter.connect() is True
        assert adapter.is_connected is True
        assert adapter.source_name == "test"


class TestCSVDataAdapter:
    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_option_data(self):
        return pd.DataFrame(
            {
                "strike": [445, 450, 455, 445, 450, 455],
                "expiration": ["2024-03-15"] * 6,
                "call_put": ["call", "call", "call", "put", "put", "put"],
                "bid": [10.5, 5.5, 2.5, 2.0, 5.0, 9.5],
                "ask": [10.7, 5.7, 2.7, 2.2, 5.2, 9.7],
                "vol": [0.20, 0.18, 0.19, 0.21, 0.19, 0.20],
                "delta": [0.75, 0.50, 0.25, -0.25, -0.50, -0.75],
                "gamma": [0.02, 0.03, 0.02, 0.02, 0.03, 0.02],
                "theta": [-0.05, -0.06, -0.04, -0.04, -0.06, -0.05],
                "vega": [0.15, 0.20, 0.15, 0.15, 0.20, 0.15],
                "rho": [0.10, 0.08, 0.05, -0.05, -0.08, -0.10],
            }
        )

    @pytest.fixture
    def nested_data_dir(self, temp_data_dir, sample_option_data):
        spy_dir = temp_data_dir / "SPY"
        spy_dir.mkdir()
        sample_option_data.to_csv(spy_dir / "2024-01-15.csv", index=False)
        sample_option_data.to_csv(spy_dir / "2024-01-16.csv", index=False)
        return temp_data_dir

    @pytest.fixture
    def flat_data_dir(self, temp_data_dir, sample_option_data):
        sample_option_data.to_csv(temp_data_dir / "SPY_20240115.csv", index=False)
        sample_option_data.to_csv(temp_data_dir / "SPY_20240116.csv", index=False)
        return temp_data_dir

    def test_connect_nonexistent_directory(self):
        adapter = CSVDataAdapter("/nonexistent/path")
        assert adapter.connect() is False
        assert adapter.is_connected is False

    def test_connect_empty_directory(self, temp_data_dir):
        adapter = CSVDataAdapter(str(temp_data_dir))
        assert adapter.connect() is False

    def test_connect_nested_structure(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir))
        assert adapter.connect() is True
        assert adapter.is_connected is True
        assert adapter._detected_pattern == "nested"

    def test_connect_flat_structure(self, flat_data_dir):
        adapter = CSVDataAdapter(str(flat_data_dir))
        assert adapter.connect() is True
        assert adapter._detected_pattern == "flat"

    def test_disconnect_clears_cache(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir))
        adapter.connect()
        adapter._cache["test"] = pd.DataFrame()
        adapter.disconnect()
        assert len(adapter._cache) == 0
        assert adapter.is_connected is False

    def test_source_name(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir))
        assert nested_data_dir.name in adapter.source_name
        assert "CSV:" in adapter.source_name

    def test_get_option_chain_nested(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir))
        adapter.connect()
        chain = adapter.get_option_chain("SPY", datetime(2024, 1, 15))
        assert len(chain) == 6
        assert "strike" in chain.columns
        assert "call_put" in chain.columns

    def test_get_option_chain_flat(self, flat_data_dir):
        adapter = CSVDataAdapter(str(flat_data_dir))
        adapter.connect()
        chain = adapter.get_option_chain("SPY", datetime(2024, 1, 15))
        assert len(chain) == 6

    def test_get_option_chain_not_connected(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir))
        with pytest.raises(DataSourceError, match="not connected"):
            adapter.get_option_chain("SPY", datetime(2024, 1, 15))

    def test_get_option_chain_not_found(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir))
        adapter.connect()
        with pytest.raises(DataNotFoundError):
            adapter.get_option_chain("SPY", datetime(2024, 12, 31))

    def test_caching_enabled(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir), use_cache=True)
        adapter.connect()
        adapter.get_option_chain("SPY", datetime(2024, 1, 15))
        assert len(adapter._cache) == 1

    def test_caching_disabled(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir), use_cache=False)
        adapter.connect()
        adapter.get_option_chain("SPY", datetime(2024, 1, 15))
        assert len(adapter._cache) == 0

    def test_get_underlying_price_from_chain(self, nested_data_dir):
        adapter = CSVDataAdapter(str(nested_data_dir))
        adapter.connect()
        price = adapter.get_underlying_price("SPY", datetime(2024, 1, 15))
        assert price == 450.0

    def test_get_underlying_price_from_file(self, temp_data_dir, sample_option_data):
        spy_dir = temp_data_dir / "SPY"
        spy_dir.mkdir()
        sample_option_data.to_csv(spy_dir / "2024-01-15.csv", index=False)

        prices_df = pd.DataFrame(
            {
                "date": ["2024-01-15", "2024-01-16"],
                "close": [448.50, 449.25],
            }
        )
        prices_df.to_csv(temp_data_dir / "SPY_prices.csv", index=False)

        adapter = CSVDataAdapter(str(temp_data_dir))
        adapter.connect()
        price = adapter.get_underlying_price("SPY", datetime(2024, 1, 15))
        assert price == 448.50


class TestDoltDataAdapter:
    @pytest.fixture
    def mock_dolt_adapter(self):
        with patch(
            "backtester.data.data_manager.DoltDataAdapter.connect"
        ) as mock_connect:
            mock_connect.return_value = True
            adapter = DoltDataAdapter("/fake/path")
            adapter._is_connected = True
            adapter._adapter = Mock()
            yield adapter

    def test_connect_failure(self):
        adapter = DoltDataAdapter("/nonexistent/path")
        with patch.object(adapter, "connect", return_value=False):
            result = adapter.connect()
            assert result is False

    def test_disconnect(self, mock_dolt_adapter):
        mock_dolt_adapter.disconnect()
        assert mock_dolt_adapter._adapter is None
        assert mock_dolt_adapter._is_connected is False

    def test_source_name(self):
        adapter = DoltDataAdapter("/path/to/options")
        assert "Dolt:" in adapter.source_name

    def test_get_option_chain_not_connected(self):
        adapter = DoltDataAdapter("/fake/path")
        with pytest.raises(DataSourceError, match="not connected"):
            adapter.get_option_chain("SPY", datetime(2024, 1, 15))

    def test_get_option_chain_success(self, mock_dolt_adapter):
        mock_df = pd.DataFrame(
            {
                "strike": [445, 450, 455],
                "call_put": ["call", "call", "call"],
                "bid": [10.0, 5.0, 2.0],
                "ask": [10.2, 5.2, 2.2],
            }
        )
        mock_dolt_adapter._adapter.get_option_chain.return_value = mock_df

        result = mock_dolt_adapter.get_option_chain("SPY", datetime(2024, 1, 15))
        assert len(result) == 3
        assert "strike" in result.columns

    def test_get_option_chain_not_found(self, mock_dolt_adapter):
        mock_dolt_adapter._adapter.get_option_chain.return_value = pd.DataFrame()

        with pytest.raises(DataNotFoundError):
            mock_dolt_adapter.get_option_chain("XYZ", datetime(2024, 1, 15))

    def test_get_underlying_price(self, mock_dolt_adapter):
        mock_chain = pd.DataFrame(
            {
                "strike": [445, 450, 455, 460, 465],
            }
        )
        mock_dolt_adapter._adapter.get_option_chain.return_value = mock_chain

        price = mock_dolt_adapter.get_underlying_price("SPY", datetime(2024, 1, 15))
        assert price == 455.0

    def test_get_volatility_data(self, mock_dolt_adapter):
        mock_vol = pd.DataFrame(
            {
                "iv_current": [0.20],
                "hv_current": [0.18],
                "iv_year_high": [0.35],
                "iv_year_low": [0.12],
            }
        )
        mock_dolt_adapter._adapter.get_underlying_prices.return_value = mock_vol

        vol_data = mock_dolt_adapter.get_volatility_data("SPY", datetime(2024, 1, 15))
        assert "iv_current" in vol_data
        assert vol_data["iv_current"] == 0.20


class TestDataSourceRegistry:
    @pytest.fixture
    def mock_adapter_success(self):
        adapter = Mock(spec=BaseDataAdapter)
        adapter.is_connected = True
        adapter.source_name = "MockSuccess"
        adapter.connect.return_value = True
        adapter.get_option_chain.return_value = pd.DataFrame(
            {
                "strike": [450],
                "call_put": ["call"],
            }
        )
        adapter.get_underlying_price.return_value = 450.0
        return adapter

    @pytest.fixture
    def mock_adapter_failure(self):
        adapter = Mock(spec=BaseDataAdapter)
        adapter.is_connected = True
        adapter.source_name = "MockFailure"
        adapter.connect.return_value = True
        adapter.get_option_chain.side_effect = DataNotFoundError("No data")
        adapter.get_underlying_price.side_effect = DataNotFoundError("No price")
        return adapter

    def test_empty_registry(self):
        registry = DataSourceRegistry()
        assert len(registry.sources) == 0
        assert len(registry.list_sources()) == 0

    def test_register_source(self, mock_adapter_success):
        registry = DataSourceRegistry()
        result = registry.register_source("test", mock_adapter_success, priority=5)
        assert result is True
        assert "test" in registry.sources
        assert len(registry.list_sources()) == 1

    def test_register_multiple_sources(
        self, mock_adapter_success, mock_adapter_failure
    ):
        registry = DataSourceRegistry()
        registry.register_source("primary", mock_adapter_success, priority=10)
        registry.register_source("backup", mock_adapter_failure, priority=5)

        sources = registry.list_sources()
        assert len(sources) == 2
        assert sources[0]["name"] == "primary"
        assert sources[1]["name"] == "backup"

    def test_unregister_source(self, mock_adapter_success):
        registry = DataSourceRegistry()
        registry.register_source("test", mock_adapter_success)
        result = registry.unregister_source("test")
        assert result is True
        assert "test" not in registry.sources

    def test_unregister_nonexistent(self):
        registry = DataSourceRegistry()
        result = registry.unregister_source("nonexistent")
        assert result is False

    def test_get_source(self, mock_adapter_success):
        registry = DataSourceRegistry()
        registry.register_source("test", mock_adapter_success)
        adapter = registry.get_source("test")
        assert adapter is mock_adapter_success

    def test_get_option_chain_success(self, mock_adapter_success):
        registry = DataSourceRegistry()
        registry.register_source("test", mock_adapter_success)

        chain = registry.get_option_chain("SPY", datetime(2024, 1, 15))
        assert len(chain) == 1
        assert chain.iloc[0]["strike"] == 450

    def test_get_option_chain_fallback(
        self, mock_adapter_success, mock_adapter_failure
    ):
        registry = DataSourceRegistry()
        registry.register_source("primary", mock_adapter_failure, priority=10)
        registry.register_source("backup", mock_adapter_success, priority=5)

        chain = registry.get_option_chain("SPY", datetime(2024, 1, 15))
        assert len(chain) == 1

    def test_get_option_chain_all_fail(self, mock_adapter_failure):
        registry = DataSourceRegistry()
        registry.register_source("test", mock_adapter_failure)

        with pytest.raises(AllSourcesFailedError):
            registry.get_option_chain("SPY", datetime(2024, 1, 15))

    def test_get_option_chain_specific_source(self, mock_adapter_success):
        registry = DataSourceRegistry()
        registry.register_source("test", mock_adapter_success)

        chain = registry.get_option_chain("SPY", datetime(2024, 1, 15), source="test")
        assert len(chain) == 1

    def test_get_option_chain_unknown_source(self, mock_adapter_success):
        registry = DataSourceRegistry()
        registry.register_source("test", mock_adapter_success)

        with pytest.raises(DataSourceError, match="Unknown source"):
            registry.get_option_chain("SPY", datetime(2024, 1, 15), source="unknown")

    def test_get_underlying_price_success(self, mock_adapter_success):
        registry = DataSourceRegistry()
        registry.register_source("test", mock_adapter_success)

        price = registry.get_underlying_price("SPY", datetime(2024, 1, 15))
        assert price == 450.0

    def test_get_underlying_price_fallback(
        self, mock_adapter_success, mock_adapter_failure
    ):
        registry = DataSourceRegistry()
        registry.register_source("primary", mock_adapter_failure, priority=10)
        registry.register_source("backup", mock_adapter_success, priority=5)

        price = registry.get_underlying_price("SPY", datetime(2024, 1, 15))
        assert price == 450.0

    def test_statistics_tracking(self, mock_adapter_success, mock_adapter_failure):
        registry = DataSourceRegistry()
        registry.register_source("primary", mock_adapter_failure, priority=10)
        registry.register_source("backup", mock_adapter_success, priority=5)

        registry.get_option_chain("SPY", datetime(2024, 1, 15))

        stats = registry.get_statistics()
        assert stats["primary"]["failure"] == 1
        assert stats["backup"]["success"] == 1

    def test_disconnect_all(self, mock_adapter_success, mock_adapter_failure):
        registry = DataSourceRegistry()
        registry.register_source("a", mock_adapter_success)
        registry.register_source("b", mock_adapter_failure)

        registry.disconnect_all()

        mock_adapter_success.disconnect.assert_called_once()
        mock_adapter_failure.disconnect.assert_called_once()

    def test_context_manager(self, mock_adapter_success):
        with DataSourceRegistry() as registry:
            registry.register_source("test", mock_adapter_success)
            chain = registry.get_option_chain("SPY", datetime(2024, 1, 15))
            assert len(chain) == 1

        mock_adapter_success.disconnect.assert_called()


class TestIntegration:
    @pytest.fixture
    def temp_csv_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            spy_dir = tmpdir / "SPY"
            spy_dir.mkdir()

            data = pd.DataFrame(
                {
                    "strike": [445, 450, 455, 445, 450, 455],
                    "expiration": ["2024-03-15"] * 6,
                    "call_put": ["call", "call", "call", "put", "put", "put"],
                    "bid": [10.5, 5.5, 2.5, 2.0, 5.0, 9.5],
                    "ask": [10.7, 5.7, 2.7, 2.2, 5.2, 9.7],
                    "vol": [0.20, 0.18, 0.19, 0.21, 0.19, 0.20],
                }
            )
            data.to_csv(spy_dir / "2024-01-15.csv", index=False)

            yield tmpdir

    def test_csv_with_registry(self, temp_csv_data):
        csv_adapter = CSVDataAdapter(str(temp_csv_data))

        with DataSourceRegistry() as registry:
            registry.register_source("csv", csv_adapter, priority=10)

            chain = registry.get_option_chain("SPY", datetime(2024, 1, 15))
            assert len(chain) == 6
            assert "strike" in chain.columns

            price = registry.get_underlying_price("SPY", datetime(2024, 1, 15))
            assert price == 450.0

    def test_multiple_sources_priority(self, temp_csv_data):
        csv_adapter = CSVDataAdapter(str(temp_csv_data))

        mock_high_priority = Mock(spec=BaseDataAdapter)
        mock_high_priority.is_connected = True
        mock_high_priority.source_name = "HighPriority"
        mock_high_priority.connect.return_value = True
        mock_high_priority.get_option_chain.return_value = pd.DataFrame(
            {
                "strike": [999],
                "call_put": ["call"],
                "source": ["high_priority"],
            }
        )

        registry = DataSourceRegistry()
        registry.register_source("csv", csv_adapter, priority=5)
        registry.register_source("high", mock_high_priority, priority=10)

        chain = registry.get_option_chain("SPY", datetime(2024, 1, 15))
        assert chain.iloc[0]["strike"] == 999
