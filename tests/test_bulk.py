"""Tests for bulk processing functionality."""

import pytest
import time
import osrm
import constants

ch_data_path = constants.ch_data_path

# Test if polars is available
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


class TestBulkRoute:
    def setup_method(self):
        self.py_osrm = osrm.OSRM(
            storage_config=ch_data_path,
            use_shared_memory=False
        )
        
        # Sample OD pairs around Monaco
        self.test_coords = [
            (7.41337, 43.72956, 7.41546, 43.73077),
            (7.41862, 43.73216, 7.42000, 43.73300),
            (7.42150, 43.73400, 7.42300, 43.73500),
            (7.41500, 43.73000, 7.42500, 43.73600),
            (7.41400, 43.72900, 7.42400, 43.73550),
        ]
    
    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_bulk_route_basic_polars(self):
        """Test basic bulk_route with Polars DataFrame."""
        df = pl.DataFrame({
            "origin_lon": [c[0] for c in self.test_coords],
            "origin_lat": [c[1] for c in self.test_coords],
            "dest_lon": [c[2] for c in self.test_coords],
            "dest_lat": [c[3] for c in self.test_coords],
        })
        
        results = osrm.bulk_route(self.py_osrm, df)
        
        assert isinstance(results, pl.DataFrame)
        assert len(results) == len(self.test_coords)
        assert "distance" in results.columns
        assert "duration" in results.columns
        assert "success" in results.columns
        assert "error" in results.columns
        
        # All routes should succeed
        assert results["success"].all()
        assert results["distance"].null_count() == 0
        assert results["duration"].null_count() == 0
    
    def test_bulk_route_dict_input(self):
        """Test bulk_route with dict-of-lists input (no Polars required)."""
        df = {
            "origin_lon": [c[0] for c in self.test_coords],
            "origin_lat": [c[1] for c in self.test_coords],
            "dest_lon": [c[2] for c in self.test_coords],
            "dest_lat": [c[3] for c in self.test_coords],
        }
        
        results = osrm.bulk_route(self.py_osrm, df)
        
        assert isinstance(results, dict)
        assert len(results["distance"]) == len(self.test_coords)
        assert len(results["duration"]) == len(self.test_coords)
        
        # All routes should succeed
        assert all(results["success"])
        assert all(d is not None for d in results["distance"])
        assert all(d is not None for d in results["duration"])
    
    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_bulk_route_with_parameters(self):
        """Test bulk_route with additional parameters."""
        df = pl.DataFrame({
            "origin_lon": [c[0] for c in self.test_coords[:3]],
            "origin_lat": [c[1] for c in self.test_coords[:3]],
            "dest_lon": [c[2] for c in self.test_coords[:3]],
            "dest_lat": [c[3] for c in self.test_coords[:3]],
        })
        
        results = osrm.bulk_route(
            self.py_osrm, 
            df, 
            steps=True,
            geometries="geojson"
        )
        
        assert results["success"].all()
        # Geometry should be present (as GeoJSON structure)
        assert results["geometry"].null_count() == 0
    
    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_bulk_route_mixed_success_failure(self):
        """Test bulk_route with some invalid coordinates."""
        df = pl.DataFrame({
            "origin_lon": [7.41337, 7.41862, 999.0, 7.41500],  # 999.0 is invalid
            "origin_lat": [43.72956, 43.73216, 43.73400, 43.73000],
            "dest_lon": [7.41546, 7.42000, 7.42300, 7.42500],
            "dest_lat": [43.73077, 43.73300, 43.73500, 43.73600],
        })
        
        results = osrm.bulk_route(self.py_osrm, df, fail_fast=False)
        
        assert len(results) == 4
        # Some should succeed, at least one should fail
        success_count = results["success"].sum()
        assert success_count >= 2  # At least the first two should work
        assert success_count < 4   # The invalid one should fail
        
        # Failed entries should have error messages
        failed = results.filter(pl.col("success") == False)
        assert len(failed) > 0
        assert failed["error"][0] is not None
    
    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_bulk_route_performance(self):
        """Test that bulk_route provides parallel speedup."""
        # Create more test data for meaningful comparison
        large_coords = self.test_coords * 20  # 100 routes for better timing resolution
        
        df = pl.DataFrame({
            "origin_lon": [c[0] for c in large_coords],
            "origin_lat": [c[1] for c in large_coords],
            "dest_lon": [c[2] for c in large_coords],
            "dest_lat": [c[3] for c in large_coords],
        })
        
        # Time sequential processing
        start = time.time()
        for _, row in enumerate(df.to_dicts()):
            self.py_osrm.Route(
                coordinates=[
                    (row["origin_lon"], row["origin_lat"]),
                    (row["dest_lon"], row["dest_lat"])
                ]
            )
        sequential_time = time.time() - start
        
        # Time parallel processing
        start = time.time()
        results = osrm.bulk_route(self.py_osrm, df, max_workers=4)
        parallel_time = time.time() - start
        
        # Parallel should be significantly faster (at least 1.3x on 4 cores)
        # Being conservative since CI environments may have limited resources
        speedup = sequential_time / parallel_time
        print(f"\nSpeedup: {speedup:.2f}x (sequential: {sequential_time:.2f}s, parallel: {parallel_time:.2f}s)")
        
        # Skip speedup assertion if routes are too fast (< 0.1s total)
        if sequential_time < 0.1:
            pytest.skip("Routes too fast to measure speedup reliably")
        
        assert speedup > 1.3, f"Expected speedup > 1.3x, got {speedup:.2f}x"
        assert results["success"].all()
    
    def test_bulk_route_fail_fast(self):
        """Test fail_fast parameter."""
        df = {
            "origin_lon": [7.41337, 999.0, 7.41862],  # Invalid coord in middle
            "origin_lat": [43.72956, 43.73216, 43.73216],
            "dest_lon": [7.41546, 7.42000, 7.42000],
            "dest_lat": [43.73077, 43.73300, 43.73300],
        }
        
        # With fail_fast=True, should raise exception
        with pytest.raises(Exception):
            osrm.bulk_route(self.py_osrm, df, fail_fast=True)
        
        # With fail_fast=False, should complete with errors
        results = osrm.bulk_route(self.py_osrm, df, fail_fast=False)
        assert len(results["success"]) == 3
    
    def test_bulk_route_max_workers(self):
        """Test max_workers parameter."""
        df = {
            "origin_lon": [c[0] for c in self.test_coords[:3]],
            "origin_lat": [c[1] for c in self.test_coords[:3]],
            "dest_lon": [c[2] for c in self.test_coords[:3]],
            "dest_lat": [c[3] for c in self.test_coords[:3]],
        }
        
        # Should work with different worker counts
        results1 = osrm.bulk_route(self.py_osrm, df, max_workers=1)
        results2 = osrm.bulk_route(self.py_osrm, df, max_workers=4)
        
        assert all(results1["success"])
        assert all(results2["success"])
        # Results should be deterministic regardless of worker count
        assert results1["distance"] == results2["distance"]
    
    def test_bulk_route_missing_columns(self):
        """Test error handling for missing required columns."""
        df = {
            "origin_lon": [7.41337],
            "origin_lat": [43.72956],
            # Missing dest_lon and dest_lat
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            osrm.bulk_route(self.py_osrm, df)
    
    def test_bulk_route_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="must be a Polars DataFrame or dict-of-lists"):
            osrm.bulk_route(self.py_osrm, "invalid input")
    
    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_bulk_route_per_row_parameters(self):
        """Test per-row parameter variations."""
        df = pl.DataFrame({
            "origin_lon": [7.41337, 7.41862],
            "origin_lat": [43.72956, 43.73216],
            "dest_lon": [7.41546, 7.42000],
            "dest_lat": [43.73077, 43.73300],
            "steps": [True, False],  # Different steps setting per row
        })
        
        results = osrm.bulk_route(self.py_osrm, df)
        
        assert results["success"].all()
        # Both should have valid results
        assert results["distance"].null_count() == 0


class TestBulkTable:
    def setup_method(self):
        self.py_osrm = osrm.OSRM(
            storage_config=ch_data_path,
            use_shared_memory=False
        )
    
    def test_bulk_table_not_implemented(self):
        """Test that bulk_table raises NotImplementedError."""
        df = {
            "locations": [
                [(7.41337, 43.72956), (7.41546, 43.73077)],
            ]
        }
        
        with pytest.raises(NotImplementedError):
            osrm.bulk_table(self.py_osrm, df)
