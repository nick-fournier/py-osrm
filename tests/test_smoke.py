"""Smoke tests for py-osrm - basic functionality without requiring OSRM data files."""
import osrm


class TestImport:
    """Test that the module imports and basic classes are available."""
    
    def test_module_import(self):
        """Test that osrm module can be imported."""
        assert osrm is not None
        
    def test_osrm_class_exists(self):
        """Test that OSRM class is available."""
        assert hasattr(osrm, 'OSRM')
        
    def test_parameter_classes_exist(self):
        """Test that all parameter classes are available."""
        assert hasattr(osrm, 'RouteParameters')
        assert hasattr(osrm, 'TableParameters')
        assert hasattr(osrm, 'MatchParameters')
        assert hasattr(osrm, 'TripParameters')
        assert hasattr(osrm, 'NearestParameters')
        assert hasattr(osrm, 'TileParameters')
        
    def test_config_classes_exist(self):
        """Test that configuration classes are available."""
        assert hasattr(osrm, 'EngineConfig')
        
    def test_enum_types_exist(self):
        """Test that enum types are available."""
        assert hasattr(osrm, 'Algorithm')
        
    def test_json_types_exist(self):
        """Test that JSON container types are available."""
        assert hasattr(osrm, 'Object')
        assert hasattr(osrm, 'Array')


class TestEnums:
    """Test enum values are properly defined."""
    
    def test_algorithm_enum_values(self):
        """Test Algorithm enum has expected values."""
        assert hasattr(osrm.Algorithm, 'CH')
        assert hasattr(osrm.Algorithm, 'MLD')
        # CoreCH was removed in OSRM 6.0.0
        assert not hasattr(osrm.Algorithm, 'CoreCH')
        

class TestParameterInstantiation:
    """Test that parameter objects can be instantiated."""
    
    def test_route_parameters_creation(self):
        """Test RouteParameters can be created with default constructor."""
        params = osrm.RouteParameters()
        assert params is not None
        
    def test_table_parameters_creation(self):
        """Test TableParameters can be created with default constructor."""
        params = osrm.TableParameters()
        assert params is not None
        
    def test_match_parameters_creation(self):
        """Test MatchParameters can be created with default constructor."""
        params = osrm.MatchParameters()
        assert params is not None
        
    def test_trip_parameters_creation(self):
        """Test TripParameters can be created with default constructor."""
        params = osrm.TripParameters()
        assert params is not None
        
    def test_nearest_parameters_creation(self):
        """Test NearestParameters can be created with default constructor."""
        params = osrm.NearestParameters()
        assert params is not None
        
    def test_coordinate_assignment(self):
        """Test that coordinates can be assigned to parameters."""
        params = osrm.RouteParameters()
        coord1 = osrm.Coordinate((7.419758, 43.731142))
        coord2 = osrm.Coordinate((7.419505, 43.736825))
        params.coordinates = [coord1, coord2]
        assert len(params.coordinates) == 2
        
    def test_tile_parameters_creation(self):
        """Test TileParameters can be created."""
        params = osrm.TileParameters([17059, 11948, 15])
        assert params is not None


class TestEngineConfig:
    """Test EngineConfig functionality."""
    
    def test_engine_config_creation(self):
        """Test EngineConfig can be instantiated."""
        config = osrm.EngineConfig()
        assert config is not None
        
    def test_algorithm_assignment(self):
        """Test Algorithm can be assigned to config."""
        config = osrm.EngineConfig()
        config.algorithm = osrm.Algorithm.MLD
        # No assertion needed - just shouldn't raise an exception
        
    def test_use_shared_memory(self):
        """Test use_shared_memory can be set."""
        config = osrm.EngineConfig()
        config.use_shared_memory = False
        # No assertion needed - just shouldn't raise an exception


class TestJSONContainers:
    """Test JSON container types."""
    
    def test_object_creation(self):
        """Test Object can be created."""
        obj = osrm.Object()
        assert obj is not None
        assert len(obj) == 0
        
    def test_array_creation(self):
        """Test Array can be created."""
        arr = osrm.Array()
        assert arr is not None
        assert len(arr) == 0


class TestProfileFiles:
    """Test that profile files are properly installed."""
    
    def test_profiles_directory_exists(self):
        """Test that profiles directory exists in installed package."""
        from pathlib import Path
        
        # Get package directory
        package_dir = Path(osrm.__file__).parent
        profiles_dir = package_dir / 'profiles'
        
        assert profiles_dir.exists(), f"Profiles directory not found: {profiles_dir}"
        assert profiles_dir.is_dir(), f"Profiles path is not a directory: {profiles_dir}"
    
    def test_bundled_profiles_exist(self):
        """Test that all bundled profile files exist."""
        from pathlib import Path
        
        package_dir = Path(osrm.__file__).parent
        profiles_dir = package_dir / 'profiles'
        
        # Check for the three main profile files
        required_profiles = ['car.lua', 'bicycle.lua', 'foot.lua']
        
        for profile_name in required_profiles:
            profile_path = profiles_dir / profile_name
            assert profile_path.exists(), f"Profile file not found: {profile_path}"
            assert profile_path.is_file(), f"Profile path is not a file: {profile_path}"
    
    def test_profile_lib_directory_exists(self):
        """Test that the lib subdirectory with shared Lua modules exists."""
        from pathlib import Path
        
        package_dir = Path(osrm.__file__).parent
        lib_dir = package_dir / 'profiles' / 'lib'
        
        assert lib_dir.exists(), f"Profiles lib directory not found: {lib_dir}"
        assert lib_dir.is_dir(), f"Profiles lib path is not a directory: {lib_dir}"
        
        # Check for some key lib files
        key_lib_files = ['utils.lua', 'sequence.lua', 'set.lua', 'way_handlers.lua']
        for lib_file in key_lib_files:
            lib_path = lib_dir / lib_file
            assert lib_path.exists(), f"Library file not found: {lib_path}"
    
    def test_profile_resolution_works(self):
        """Test that the internal _resolve_profile function works correctly."""
        from osrm.preprocessing import _resolve_profile
        
        # Test with bundled profile names
        for profile_name in ['car', 'bicycle', 'foot']:
            profile_path = _resolve_profile(profile_name)
            assert profile_path.exists(), f"Resolved profile doesn't exist: {profile_path}"
            assert profile_path.suffix == '.lua', f"Resolved profile is not a .lua file: {profile_path}"
    
    def test_profile_resolution_returns_absolute_paths(self):
        """Test that _resolve_profile always returns absolute paths."""
        from osrm.preprocessing import _resolve_profile
        
        # Test with bundled profile names
        for profile_name in ['car', 'bicycle', 'foot']:
            profile_path = _resolve_profile(profile_name)
            assert profile_path.is_absolute(), f"Profile path is not absolute: {profile_path}"
    
    def test_profile_works_with_extractor_config(self):
        """Test that resolved profile can be assigned to ExtractorConfig."""
        from osrm.preprocessing import _resolve_profile
        import osrm
        from pathlib import Path
        
        # Create an ExtractorConfig
        config = osrm.ExtractorConfig()
        
        # Test assigning resolved profile paths
        for profile_name in ['car', 'bicycle', 'foot']:
            profile_path = _resolve_profile(profile_name)
            # Convert to string as done in extract() function
            config.profile_path = str(profile_path.absolute())
            
            # Verify it was set (should not raise an exception)
            assert config.profile_path is not None
