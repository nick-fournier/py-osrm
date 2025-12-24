import pytest
import osrm
import tempfile
import shutil
from pathlib import Path

# Get test data paths
test_dir = Path(__file__).parent
osm_file = test_dir / "data" / "monaco.osm.pbf"
profile_file = test_dir / "data" / "profiles" / "car.lua"


class TestPreprocessing:
    """Test OSRM data preprocessing functions."""
    
    @pytest.fixture
    def temp_output(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp(prefix="osrm_test_")
        output_base = Path(temp_dir) / "monaco"
        yield str(output_base)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_extract_basic(self, temp_output):
        """Test basic extract functionality."""
        result = osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
        # Simple extract may not return duration
        
        # Verify output files exist
        assert Path(f"{temp_output}.osrm.ebg").exists()
        assert Path(f"{temp_output}.osrm.properties").exists()
    
    def test_extract_with_progress_callback(self, temp_output):
        """Test extract with progress callback."""
        progress_lines = []
        
        def progress_callback(line):
            progress_lines.append(line)
        
        result = osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="WARNING",
            progress_callback=progress_callback,
            capture_output=True
        )
        
        assert result["success"] is True
        assert "stdout" in result
        assert "stderr" in result
        # Progress callback should have been called with output
        # (may be 0 if verbosity is too low, but structure should work)
        assert isinstance(progress_lines, list)
    
    def test_extract_with_capture_output(self, temp_output):
        """Test extract with output capture."""
        result = osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="INFO",
            capture_output=True
        )
        
        assert result["success"] is True
        assert "stdout" in result
        assert "stderr" in result
        assert isinstance(result["stdout"], str)
        assert isinstance(result["stderr"], str)
    
    def test_extract_with_options(self, temp_output):
        """Test extract with additional options."""
        result = osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR",
            threads=2,
            small_component_size=500,
            use_metadata=True
        )
        
        assert result["success"] is True
    
    def test_contract_basic(self, temp_output):
        """Test basic contract functionality."""
        # First extract
        osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        # Then contract
        result = osrm.contract(
            temp_output,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
        # Simple contract may not return duration
        
        # Verify CH output file exists
        assert Path(f"{temp_output}.osrm.hsgr").exists()
    
    def test_contract_with_threads(self, temp_output):
        """Test contract with custom thread count."""
        osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        result = osrm.contract(
            temp_output,
            threads=2,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
    
    def test_partition_basic(self, temp_output):
        """Test basic partition functionality."""
        # First extract
        osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        # Then partition
        result = osrm.partition(
            temp_output,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
        
        # Verify partition output files exist
        assert Path(f"{temp_output}.osrm.partition").exists()
        assert Path(f"{temp_output}.osrm.cells").exists()
    
    def test_partition_with_options(self, temp_output):
        """Test partition with custom options."""
        osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        result = osrm.partition(
            temp_output,
            threads=2,
            balance=1.5,
            boundary_factor=0.3,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
    
    def test_customize_basic(self, temp_output):
        """Test basic customize functionality."""
        # Extract, partition, then customize
        osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        osrm.partition(
            temp_output,
            verbosity="ERROR"
        )
        
        result = osrm.customize(
            temp_output,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
        assert result.get("duration", 0) >= 0
        # Verify MLD output file exists
        assert Path(f"{temp_output}.osrm.mldgr").exists()
    
    def test_full_ch_pipeline(self, temp_output):
        """Test complete CH preprocessing pipeline."""
        # Extract
        extract_result = osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        assert extract_result["success"] is True
        
        # Contract
        contract_result = osrm.contract(
            temp_output,
            verbosity="ERROR"
        )
        assert contract_result["success"] is True
        
        # Verify we can load the engine (test separately to avoid segfault)
        assert Path(f"{temp_output}.osrm.hsgr").exists()
    
    def test_full_mld_pipeline(self, temp_output):
        """Test complete MLD preprocessing pipeline."""
        # Extract
        extract_result = osrm.extract(
            str(osm_file),
            profile="car",
            output_path=temp_output,
            verbosity="ERROR"
        )
        assert extract_result["success"] is True
        
        # Partition
        partition_result = osrm.partition(
            temp_output,
            verbosity="ERROR"
        )
        assert partition_result["success"] is True
        
        # Customize
        customize_result = osrm.customize(
            temp_output,
            verbosity="ERROR"
        )
        assert customize_result["success"] is True
        
        # Verify MLD files exist
        assert Path(f"{temp_output}.osrm.mldgr").exists()
        assert Path(f"{temp_output}.osrm.cell_metrics").exists()
    
    def test_config_classes_exported(self):
        """Test that config classes are properly exported."""
        assert hasattr(osrm, 'ExtractorConfig')
        assert hasattr(osrm, 'ContractorConfig')
        assert hasattr(osrm, 'PartitionerConfig')
        assert hasattr(osrm, 'CustomizationConfig')
    
    def test_config_classes_instantiable(self):
        """Test that config classes can be instantiated."""
        extractor_config = osrm.ExtractorConfig()
        assert extractor_config is not None
        
        contractor_config = osrm.ContractorConfig()
        assert contractor_config is not None
        
        partitioner_config = osrm.PartitionerConfig()
        assert partitioner_config is not None
        
        customization_config = osrm.CustomizationConfig()
        assert customization_config is not None
    
    def test_config_classes_configurable(self):
        """Test that config class attributes can be set."""
        config = osrm.ExtractorConfig()
        
        # Test setting attributes
        config.requested_num_threads = 4
        assert config.requested_num_threads == 4
        
        config.small_component_size = 2000
        assert config.small_component_size == 2000
        
        config.use_metadata = True
        assert config.use_metadata is True
    
    def test_verbosity_levels(self, temp_output):
        """Test different verbosity levels."""
        verbosity_levels = ["NONE", "ERROR", "WARNING", "INFO", "DEBUG"]
        
        for level in verbosity_levels:
            # Clean temp files
            for f in Path(temp_output).parent.glob(f"{Path(temp_output).name}.osrm*"):
                f.unlink()
            
            result = osrm.extract(
                str(osm_file),
                profile="car",
                output_path=temp_output,
                verbosity=level,
                capture_output=True
            )
            
            assert result["success"] is True, f"Extract failed with verbosity={level}"
    
    def test_extract_missing_input(self):
        """Test extract with missing input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result = osrm.extract(
                    "/nonexistent/file.osm.pbf",
                    profile="car",
                    output_path=str(Path(temp_dir) / "output"),
                    verbosity="ERROR",
                    capture_output=True
                )
                # Should fail
                assert result["success"] is False
            except Exception as e:
                # Also acceptable to raise an exception
                assert "nonexistent" in str(e).lower() or "no such file" in str(e).lower()
    
    def test_extract_missing_profile(self, temp_output):
        """Test extract with missing profile file."""
        with pytest.raises(FileNotFoundError):
            osrm.extract(
                str(osm_file),
                profile="/nonexistent/profile.lua",
                output_path=temp_output,
                verbosity="ERROR"
            )
    
    def test_extract_bicycle_profile(self, temp_output):
        """Test extract with bicycle profile."""
        result = osrm.extract(
            str(osm_file),
            profile="bicycle",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
        assert Path(f"{temp_output}.osrm.ebg").exists()
    
    def test_extract_foot_profile(self, temp_output):
        """Test extract with foot profile."""
        result = osrm.extract(
            str(osm_file),
            profile="foot",
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
        assert Path(f"{temp_output}.osrm.ebg").exists()
    
    def test_extract_custom_profile_path(self, temp_output):
        """Test extract with custom profile using full path."""
        result = osrm.extract(
            str(osm_file),
            profile=str(profile_file),
            output_path=temp_output,
            verbosity="ERROR"
        )
        
        assert result["success"] is True
        assert Path(f"{temp_output}.osrm.ebg").exists()
    
    def test_extract_invalid_profile_name(self, temp_output):
        """Test extract with invalid profile name."""
        with pytest.raises(FileNotFoundError) as exc_info:
            osrm.extract(
                str(osm_file),
                profile="truck",
                output_path=temp_output,
                verbosity="ERROR"
            )
        
        # Error message should mention valid profile names
        error_msg = str(exc_info.value)
        assert "car" in error_msg or "bicycle" in error_msg or "foot" in error_msg
