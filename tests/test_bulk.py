"""Tests for bulk processing functionality."""

import pytest
import time
import osrm
import constants
import polars as pl

ch_data_path = constants.ch_data_path


class TestBulkRoute:
    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config=ch_data_path,
            use_shared_memory=False
        )
        
        # Sample OD pairs around Monaco
        cls.test_coords = [
            (7.41337, 43.72956, 7.41546, 43.73077),
            (7.41862, 43.73216, 7.42000, 43.73300),
            (7.42150, 43.73400, 7.42300, 43.73500),
            (7.41500, 43.73000, 7.42500, 43.73600),
            (7.41400, 43.72900, 7.42400, 43.73550),
        ]
    
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
    
    def test_bulk_route_performance(self):
        """Test that bulk_route provides parallel speedup."""
        # Create enough test data for stable timing measurements
        large_coords = self.test_coords * 2000  # 5000 routes
        
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
        
        # Time parallel processing (disable progress bar for accurate timing)
        start = time.time()
        results = osrm.bulk_route(self.py_osrm, df, max_workers=4, show_progress=False)
        parallel_time = time.time() - start
        
        # Parallel should be faster with chunked dispatch on 4 cores
        speedup = sequential_time / parallel_time
        print(f"\nSpeedup: {speedup:.2f}x (sequential: {sequential_time:.2f}s, parallel: {parallel_time:.2f}s)")
        
        # Skip speedup assertion if routes are too fast to measure reliably
        if sequential_time < 0.5:
            pytest.skip("Routes too fast to measure speedup reliably")
        
        assert speedup > 1.2, f"Expected speedup > 1.2x, got {speedup:.2f}x"
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
    
    def test_bulk_route_progress_bar_default(self):
        """Test that progress bar is shown by default."""
        from unittest.mock import patch, MagicMock
        
        df = pl.DataFrame({
            "origin_lon": [c[0] for c in self.test_coords[:3]],
            "origin_lat": [c[1] for c in self.test_coords[:3]],
            "dest_lon": [c[2] for c in self.test_coords[:3]],
            "dest_lat": [c[3] for c in self.test_coords[:3]],
        })
        
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            mock_tqdm.return_value = mock_progress
            
            results = osrm.bulk_route(self.py_osrm, df)
            
            # tqdm should be called (progress bar enabled by default)
            mock_tqdm.assert_called_once()
            # Progress bar should be updated for each item
            assert mock_progress.update.call_count == 3
            # Progress bar should be closed
            mock_progress.close.assert_called_once()
            assert results["success"].all()
    
    def test_bulk_route_progress_bar_disabled(self):
        """Test that progress bar can be disabled."""
        from unittest.mock import patch
        
        df = pl.DataFrame({
            "origin_lon": [c[0] for c in self.test_coords[:3]],
            "origin_lat": [c[1] for c in self.test_coords[:3]],
            "dest_lon": [c[2] for c in self.test_coords[:3]],
            "dest_lat": [c[3] for c in self.test_coords[:3]],
        })
        
        with patch('tqdm.tqdm') as mock_tqdm:
            results = osrm.bulk_route(self.py_osrm, df, show_progress=False)
            
            # tqdm should NOT be called when show_progress=False
            mock_tqdm.assert_not_called()
            assert results["success"].all()
    
    def test_bulk_route_progress_bar_error_tracking(self):
        """Test that progress bar tracks errors."""
        from unittest.mock import patch, MagicMock
        
        df = pl.DataFrame({
            "origin_lon": [7.41337, 999.0, 7.41862],  # Invalid coord in middle
            "origin_lat": [43.72956, 43.73216, 43.73216],
            "dest_lon": [7.41546, 7.42000, 7.42000],
            "dest_lat": [43.73077, 43.73300, 43.73300],
        })
        
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            mock_tqdm.return_value = mock_progress
            
            results = osrm.bulk_route(self.py_osrm, df, fail_fast=False)
            
            # Progress bar should track errors in postfix
            # Extract error counts from all set_postfix calls
            postfix_calls = [call.args[0]['errors'] if call.args else call.kwargs.get('errors', 0) 
                           for call in mock_progress.set_postfix.call_args_list]
            # At least one call should have errors > 0 (the invalid coordinate should fail)
            assert any(errors > 0 for errors in postfix_calls), f"Error count should be tracked, got: {postfix_calls}"


class TestBulkTable:
    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config=ch_data_path,
            use_shared_memory=False
        )
    

class TestBulkNearest:
    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config=ch_data_path,
            use_shared_memory=False
        )
        
        # Sample coordinates around Monaco
        cls.test_coords = [
            (7.41337, 43.72956),
            (7.41862, 43.73216),
            (7.42150, 43.73400),
            (7.41500, 43.73000),
            (7.41400, 43.72900),
        ]
    
    def test_bulk_nearest_basic_polars(self):
        """Test basic bulk_nearest with Polars DataFrame."""
        df = pl.DataFrame({
            "lon": [c[0] for c in self.test_coords],
            "lat": [c[1] for c in self.test_coords],
        })
        
        results = osrm.bulk_nearest(self.py_osrm, df)
        
        assert isinstance(results, pl.DataFrame)
        assert len(results) == len(self.test_coords)
        assert "waypoint_lon" in results.columns
        assert "waypoint_lat" in results.columns
        assert "waypoint_name" in results.columns
        assert "distance" in results.columns
        assert "success" in results.columns
        assert "error" in results.columns
        
        # All requests should succeed
        assert results["success"].all()
        assert results["waypoint_lon"].null_count() == 0
        assert results["waypoint_lat"].null_count() == 0
    
    def test_bulk_nearest_dict_input(self):
        """Test bulk_nearest with dict-of-lists input (no Polars required)."""
        df = {
            "lon": [c[0] for c in self.test_coords],
            "lat": [c[1] for c in self.test_coords],
        }
        
        results = osrm.bulk_nearest(self.py_osrm, df)
        
        assert isinstance(results, dict)
        assert len(results["waypoint_lon"]) == len(self.test_coords)
        assert len(results["waypoint_lat"]) == len(self.test_coords)
        
        # All requests should succeed
        assert all(results["success"])
        assert all(d is not None for d in results["waypoint_lon"])
        assert all(d is not None for d in results["waypoint_lat"])
    
    def test_bulk_nearest_with_number(self):
        """Test bulk_nearest with number parameter to get multiple results."""
        df = pl.DataFrame({
            "lon": [c[0] for c in self.test_coords[:3]],
            "lat": [c[1] for c in self.test_coords[:3]],
        })
        
        results = osrm.bulk_nearest(self.py_osrm, df, number=3)
        
        assert results["success"].all()
        assert results["distance"].null_count() == 0
    
    def test_bulk_nearest_mixed_success_failure(self):
        """Test bulk_nearest with some invalid coordinates."""
        df = pl.DataFrame({
            "lon": [7.41337, 999.0, 7.41862],  # 999.0 is invalid
            "lat": [43.72956, 43.73216, 43.73216],
        })
        
        results = osrm.bulk_nearest(self.py_osrm, df, fail_fast=False)
        
        assert len(results) == 3
        # Some should succeed, at least one should fail
        success_count = results["success"].sum()
        assert success_count >= 1
        assert success_count < 3
        
        # Failed entries should have error messages
        failed = results.filter(pl.col("success") == False)
        assert len(failed) > 0
        assert failed["error"][0] is not None
    
    def test_bulk_nearest_fail_fast(self):
        """Test fail_fast parameter."""
        df = {
            "lon": [7.41337, 999.0, 7.41862],  # Invalid coord in middle
            "lat": [43.72956, 43.73216, 43.73216],
        }
        
        # With fail_fast=True, should raise exception
        with pytest.raises(Exception):
            osrm.bulk_nearest(self.py_osrm, df, fail_fast=True)
        
        # With fail_fast=False, should complete with errors
        results = osrm.bulk_nearest(self.py_osrm, df, fail_fast=False)
        assert len(results["success"]) == 3
    
    def test_bulk_nearest_max_workers(self):
        """Test max_workers parameter."""
        df = {
            "lon": [c[0] for c in self.test_coords[:3]],
            "lat": [c[1] for c in self.test_coords[:3]],
        }
        
        # Should work with different worker counts
        results1 = osrm.bulk_nearest(self.py_osrm, df, max_workers=1)
        results2 = osrm.bulk_nearest(self.py_osrm, df, max_workers=4)
        
        assert all(results1["success"])
        assert all(results2["success"])
    
    def test_bulk_nearest_missing_columns(self):
        """Test error handling for missing required columns."""
        df = {
            "lon": [7.41337],
            # Missing lat
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            osrm.bulk_nearest(self.py_osrm, df)
    
    def test_bulk_nearest_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="must be a Polars DataFrame or dict-of-lists"):
            osrm.bulk_nearest(self.py_osrm, "invalid input")
    
    def test_bulk_nearest_progress_bar_default(self):
        """Test that progress bar is shown by default."""
        from unittest.mock import patch, MagicMock
        
        df = pl.DataFrame({
            "lon": [c[0] for c in self.test_coords[:3]],
            "lat": [c[1] for c in self.test_coords[:3]],
        })
        
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            mock_tqdm.return_value = mock_progress
            
            results = osrm.bulk_nearest(self.py_osrm, df)
            
            # tqdm should be called (progress bar enabled by default)
            mock_tqdm.assert_called_once()
            # Progress bar should be updated for each item
            assert mock_progress.update.call_count == 3
            # Progress bar should be closed
            mock_progress.close.assert_called_once()
            assert results["success"].all()
    
    def test_bulk_nearest_progress_bar_disabled(self):
        """Test that progress bar can be disabled."""
        from unittest.mock import patch
        
        df = pl.DataFrame({
            "lon": [c[0] for c in self.test_coords[:3]],
            "lat": [c[1] for c in self.test_coords[:3]],
        })
        
        with patch('tqdm.tqdm') as mock_tqdm:
            results = osrm.bulk_nearest(self.py_osrm, df, show_progress=False)
            
            # tqdm should NOT be called when show_progress=False
            mock_tqdm.assert_not_called()
            assert results["success"].all()


class TestBulkMatch:
    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config=ch_data_path,
            use_shared_memory=False
        )
        
        # Sample GPS traces around Monaco
        cls.test_traces = [
            [(7.41337, 43.72956), (7.41546, 43.73077), (7.41862, 43.73216)],
            [(7.42000, 43.73300), (7.42150, 43.73400), (7.42300, 43.73500)],
            [(7.41500, 43.73000), (7.41700, 43.73100), (7.41900, 43.73200)],
        ]
    
    def test_bulk_match_basic_polars(self):
        """Test basic bulk_match with Polars DataFrame."""
        df = pl.DataFrame({
            "coordinates": self.test_traces,
        })
        
        results = osrm.bulk_match(self.py_osrm, df, fail_fast=False)
        
        assert isinstance(results, pl.DataFrame)
        assert len(results) == len(self.test_traces)
        assert "distance" in results.columns
        assert "duration" in results.columns
        assert "confidence" in results.columns
        assert "geometry" in results.columns
        assert "success" in results.columns
        assert "error" in results.columns
        
        # At least some matches should succeed
        success_count = results["success"].sum()
        assert success_count >= 1, "At least one match should succeed"
        
        # Successful matches should have valid data
        successful = results.filter(pl.col("success") == True)
        assert successful["distance"].null_count() == 0
        assert successful["duration"].null_count() == 0
    
    def test_bulk_match_dict_input(self):
        """Test bulk_match with dict-of-lists input (no Polars required)."""
        df = {
            "coordinates": self.test_traces,
        }
        
        results = osrm.bulk_match(self.py_osrm, df, fail_fast=False)
        
        assert isinstance(results, dict)
        assert len(results["distance"]) == len(self.test_traces)
        assert len(results["duration"]) == len(self.test_traces)
        assert len(results["success"]) == len(self.test_traces)
        
        # At least some matches should succeed (not all traces may be matchable)
        success_count = sum(results["success"])
        assert success_count >= 1, "At least one match should succeed"
        
        # Successful matches should have valid distance and duration
        for i, success in enumerate(results["success"]):
            if success:
                assert results["distance"][i] is not None
                assert results["duration"][i] is not None
    
    def test_bulk_match_with_parameters(self):
        """Test bulk_match with additional parameters."""
        df = pl.DataFrame({
            "coordinates": self.test_traces[:2],
        })
        
        results = osrm.bulk_match(
            self.py_osrm, 
            df, 
            steps=True,
            geometries="geojson",
            fail_fast=False
        )
        
        # At least one should succeed
        assert results["success"].sum() >= 1
        # Successful matches should have geometry present
        successful = results.filter(pl.col("success") == True)
        assert len(successful) > 0
        assert successful["geometry"].null_count() == 0
    
    def test_bulk_match_with_timestamps(self):
        """Test bulk_match with timestamps."""
        df = pl.DataFrame({
            "coordinates": [self.test_traces[0]],
            "timestamps": [[1, 2, 3]],  # Simple timestamps
        })
        
        results = osrm.bulk_match(self.py_osrm, df)
        
        assert results["success"].all()
        assert results["confidence"].null_count() == 0
    
    def test_bulk_match_mixed_success_failure(self):
        """Test bulk_match with some invalid traces."""
        df = pl.DataFrame({
            "coordinates": [
                self.test_traces[0],
                [(999.0, 43.73216), (999.1, 43.73300)],  # Invalid coords
                self.test_traces[1],
            ]
        })
        
        results = osrm.bulk_match(self.py_osrm, df, fail_fast=False)
        
        assert len(results) == 3
        # Some should succeed, at least one should fail
        success_count = results["success"].sum()
        assert success_count >= 1
        
        # Failed entries should have error messages
        failed = results.filter(pl.col("success") == False)
        if len(failed) > 0:
            assert failed["error"][0] is not None
    
    def test_bulk_match_fail_fast(self):
        """Test fail_fast parameter."""
        df = {
            "coordinates": [
                self.test_traces[0],
                [(999.0, 43.73216), (999.1, 43.73300)],  # Invalid coords
            ]
        }
        
        # With fail_fast=False, should complete with errors
        results = osrm.bulk_match(self.py_osrm, df, fail_fast=False)
        assert len(results["success"]) == 2
    
    def test_bulk_match_max_workers(self):
        """Test max_workers parameter."""
        df = {
            "coordinates": self.test_traces[:2],
        }
        
        # Should work with different worker counts
        results1 = osrm.bulk_match(self.py_osrm, df, max_workers=1, fail_fast=False)
        results2 = osrm.bulk_match(self.py_osrm, df, max_workers=4, fail_fast=False)
        
        # At least one should succeed in each
        assert sum(results1["success"]) >= 1
        assert sum(results2["success"]) >= 1
    
    def test_bulk_match_missing_columns(self):
        """Test error handling for missing required columns."""
        df = {
            "other_col": [1, 2, 3],
            # Missing coordinates
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            osrm.bulk_match(self.py_osrm, df)
    
    def test_bulk_match_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="must be a Polars DataFrame or dict-of-lists"):
            osrm.bulk_match(self.py_osrm, "invalid input")
    
    def test_bulk_match_progress_bar_default(self):
        """Test that progress bar is shown by default."""
        from unittest.mock import patch, MagicMock
        
        df = pl.DataFrame({
            "coordinates": self.test_traces[:2],
        })
        
        with patch('tqdm.tqdm') as mock_tqdm:
            mock_progress = MagicMock()
            mock_tqdm.return_value = mock_progress
            
            results = osrm.bulk_match(self.py_osrm, df, fail_fast=False)
            
            # tqdm should be called (progress bar enabled by default)
            mock_tqdm.assert_called_once()
            # Progress bar should be updated for each item
            assert mock_progress.update.call_count == 2
            # Progress bar should be closed
            mock_progress.close.assert_called_once()
    
    def test_bulk_match_progress_bar_disabled(self):
        """Test that progress bar can be disabled."""
        from unittest.mock import patch
        
        df = pl.DataFrame({
            "coordinates": self.test_traces[:2],
        })
        
        with patch('tqdm.tqdm') as mock_tqdm:
            results = osrm.bulk_match(self.py_osrm, df, show_progress=False, fail_fast=False)
            
            # tqdm should NOT be called when show_progress=False
            mock_tqdm.assert_not_called()


class TestBatchRoute:
    """Tests for the native C++ BatchRoute method."""

    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config=ch_data_path,
            use_shared_memory=False,
        )
        cls.test_coords = [
            (7.41337, 43.72956, 7.41546, 43.73077),
            (7.41862, 43.73216, 7.42000, 43.73300),
            (7.42150, 43.73400, 7.42300, 43.73500),
            (7.41500, 43.73000, 7.42500, 43.73600),
            (7.41400, 43.72900, 7.42400, 43.73550),
        ]

    def _build_params(self, coords_list):
        """Build RouteParameters list from (olon, olat, dlon, dlat) tuples."""
        return [
            osrm.RouteParameters(coordinates=[(c[0], c[1]), (c[2], c[3])])
            for c in coords_list
        ]

    def test_batch_route_matches_serial(self):
        """BatchRoute results must match serial Route for the same inputs."""
        params_list = self._build_params(self.test_coords)

        batch = self.py_osrm.BatchRoute(params_list)

        for i, (bp, coords) in enumerate(zip(batch, self.test_coords)):
            serial = self.py_osrm.Route(
                coordinates=[(coords[0], coords[1]), (coords[2], coords[3])]
            )
            assert bp is not None, f"Route {i} returned None in batch"
            assert abs(bp['routes'][0]['distance'] - serial['routes'][0]['distance']) < 1e-6
            assert abs(bp['routes'][0]['duration'] - serial['routes'][0]['duration']) < 1e-6

    def test_batch_route_empty_input(self):
        """Empty list should return empty list."""
        result = self.py_osrm.BatchRoute([])
        assert result == []

    def test_batch_route_single(self):
        """Single route batch should work."""
        params = self._build_params(self.test_coords[:1])
        result = self.py_osrm.BatchRoute(params)
        assert len(result) == 1
        assert result[0] is not None
        assert 'routes' in result[0]

    def test_batch_route_failed_routes_return_none(self):
        """Failed routes should return None without crashing the batch."""
        good = osrm.RouteParameters(coordinates=[(7.41337, 43.72956), (7.41546, 43.73077)])
        bad = osrm.RouteParameters(coordinates=[(999.0, 999.0), (999.0, 999.0)])

        result = self.py_osrm.BatchRoute([good, bad, good])
        assert len(result) == 3
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    def test_batch_route_all_fail(self):
        """All-failing batch should return all Nones."""
        bad = osrm.RouteParameters(coordinates=[(999.0, 999.0), (999.0, 999.0)])
        result = self.py_osrm.BatchRoute([bad, bad])
        assert all(r is None for r in result)

    def test_batch_route_performance(self):
        """BatchRoute should be faster than serial Route."""
        import random
        random.seed(42)
        N = 1000
        coords = [
            (random.uniform(7.408, 7.440), random.uniform(43.724, 43.752),
             random.uniform(7.408, 7.440), random.uniform(43.724, 43.752))
            for _ in range(N)
        ]
        params_list = self._build_params(coords)

        # Serial
        t0 = time.perf_counter()
        for p in params_list:
            self.py_osrm._engine.Route(p)
        serial_time = time.perf_counter() - t0

        # Batch
        t0 = time.perf_counter()
        self.py_osrm.BatchRoute(params_list)
        batch_time = time.perf_counter() - t0

        speedup = serial_time / batch_time
        if serial_time < 0.05:
            pytest.skip("Routes too fast to measure speedup reliably")
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"


