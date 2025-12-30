"""
OSRM preprocessing functions for data preparation.

This module provides Python bindings for OSRM's data preprocessing pipeline:
- extract: Parse OSM data and apply routing profile
- contract: Build contraction hierarchy (CH algorithm)
- partition: Partition graph for MLD algorithm
- customize: Customize partitioned graph for MLD algorithm
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any

from . import osrm_ext


def _resolve_profile(profile: str) -> Path:
    """
    Resolve a profile name or path to an absolute path.
    
    Args:
        profile: Either a profile name ('car', 'bicycle', 'foot') or a path to a .lua file
    
    Returns:
        Path to the profile file
    
    Raises:
        ValueError: If profile name is not recognized
        FileNotFoundError: If custom profile path doesn't exist
    """
    # Known profile names
    known_profiles = {'car', 'bicycle', 'foot'}
    
    # If it's a simple known profile name, use bundled profile
    if profile.lower() in known_profiles:
        package_dir = Path(__file__).parent
        profile_path = package_dir / 'profiles' / f'{profile.lower()}.lua'
        
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Bundled profile not found: {profile_path}. "
                "This is likely a package installation issue."
            )
        
        return profile_path.absolute()
    
    # Otherwise treat as a custom profile path
    profile_path = Path(profile)
    
    # Check if file exists
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile file not found: {profile}. "
            f"Valid profile names are: {', '.join(sorted(known_profiles))} "
            "or provide a path to a custom .lua profile file."
        )
    
    return profile_path.absolute()


def extract(
    input_path: str,
    profile: str = "car",
    output_path: Optional[str] = None,
    threads: Optional[int] = None,
    verbosity: str = "INFO",
    progress_callback: Optional[Callable[[str], None]] = None,
    capture_output: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract OSM data into OSRM format.
    
    This is the first step in the OSRM preprocessing pipeline. It parses OSM data
    (from .osm, .osm.bz2, or .osm.pbf files) and applies a Lua routing profile to
    generate graph data for routing.
    
    Args:
        input_path: Path to input OSM file (.osm, .osm.bz2, or .osm.pbf)
        profile: Profile name ('car', 'bicycle', 'foot') or path to custom .lua file
        output_path: Base path for output files. If None, uses input_path base name.
        threads: Number of threads to use. If None, uses all available CPU cores.
        verbosity: Log level - one of: "NONE", "ERROR", "WARNING", "INFO", "DEBUG"
        progress_callback: Optional callback function(line: str) for progress updates.
                          Called with each log line. Keep lightweight to avoid slowing
                          down the extraction process.
        capture_output: If True, capture stdout/stderr and return in result dict.
                       If False and no progress_callback, output goes directly to console.
        **kwargs: Additional ExtractorConfig parameters (small_component_size,
                 use_metadata, parse_conditionals, etc.)
    
    Returns:
        Dict with keys:
            - success: bool - Whether extraction succeeded
            - duration: float - Time taken in seconds
            - stdout: str - Captured stdout (if capture_output=True or progress_callback set)
            - stderr: str - Captured stderr (if capture_output=True or progress_callback set)
            - error: str - Error message (if success=False)
    
    Example:
        >>> import osrm
        >>> 
        >>> # Simple extraction with bundled profile
        >>> result = osrm.extract('data.osm.pbf', profile='car')
        >>> 
        >>> # With different profile
        >>> result = osrm.extract('data.osm.pbf', profile='bicycle')
        >>> 
        >>> # With custom profile
        >>> result = osrm.extract('data.osm.pbf', profile='path/to/custom.lua')
        >>> 
        >>> # With progress callback
        >>> def show_progress(line):
        ...     print(f"Progress: {line}")
        >>> result = osrm.extract('data.osm.pbf', progress_callback=show_progress)
        >>> print(f"Completed in {result['duration']:.2f}s")
        >>> 
        >>> # With custom settings
        >>> result = osrm.extract(
        ...     'data.osm.pbf',
        ...     profile='bicycle',
        ...     threads=4,
        ...     small_component_size=500,
        ...     use_metadata=True
        ... )
    """
    # Create config
    config = osrm_ext.ExtractorConfig()
    config.input_path = Path(input_path)
    
    # Resolve profile path - convert to absolute string for C++ binding
    profile_path = _resolve_profile(profile)
    config.profile_path = str(profile_path.absolute())
    
    # Set output path
    if output_path:
        config.UseDefaultOutputNames(Path(output_path))
    else:
        config.UseDefaultOutputNames(Path(input_path))
    
    # Set threads
    if threads is not None:
        config.requested_num_threads = threads
    else:
        # Default to CPU count to avoid TBB issues with 0 threads
        import os
        config.requested_num_threads = os.cpu_count() or 1
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Execute
    if capture_output or progress_callback is not None:
        return osrm_ext.extract_with_capture(config, verbosity, progress_callback)
    else:
        osrm_ext.extract(config, verbosity)
        return {"success": True}


def contract(
    input_path: str,
    threads: Optional[int] = None,
    verbosity: str = "INFO",
    progress_callback: Optional[Callable[[str], None]] = None,
    capture_output: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run contraction hierarchy computation for CH algorithm.
    
    This is used for the CH (Contraction Hierarchies) routing algorithm.
    Run this after extract() if you plan to use CH routing.
    
    For MLD algorithm, use partition() and customize() instead.
    
    Args:
        input_path: Base path to .osrm files (output from extract)
        threads: Number of threads to use. If None, uses all available CPU cores.
        verbosity: Log level - one of: "NONE", "ERROR", "WARNING", "INFO", "DEBUG"
        progress_callback: Optional callback function(line: str) for progress updates
        capture_output: If True, capture and return stdout/stderr
        **kwargs: Additional ContractorConfig parameters
    
    Returns:
        Dict with success, duration, stdout, stderr, and optionally error keys
    
    Example:
        >>> import osrm
        >>> 
        >>> # First extract
        >>> osrm.extract('data.osm.pbf', profile_path='profiles/car.lua')
        >>> 
        >>> # Then contract
        >>> result = osrm.contract('data.osrm')
        >>> print(f"Contraction completed in {result['duration']:.2f}s")
    """
    config = osrm_ext.ContractorConfig()
    config.UseDefaultOutputNames(Path(input_path))
    
    if threads is not None:
        config.requested_num_threads = threads
    else:
        # Default to CPU count to avoid TBB issues with 0 threads
        import os
        config.requested_num_threads = os.cpu_count() or 1
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    if capture_output or progress_callback is not None:
        return osrm_ext.contract_with_capture(config, verbosity, progress_callback)
    else:
        osrm_ext.contract(config, verbosity)
        return {"success": True}


def partition(
    input_path: str,
    threads: Optional[int] = None,
    verbosity: str = "INFO",
    progress_callback: Optional[Callable[[str], None]] = None,
    capture_output: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Partition graph for MLD (Multi-Level Dijkstra) algorithm.
    
    This is the second step for MLD routing (after extract).
    Follow with customize() to complete the MLD preprocessing pipeline.
    
    Args:
        input_path: Base path to .osrm files (output from extract)
        threads: Number of threads to use. If None, uses all available CPU cores.
        verbosity: Log level - one of: "NONE", "ERROR", "WARNING", "INFO", "DEBUG"
        progress_callback: Optional callback function(line: str) for progress updates
        capture_output: If True, capture and return stdout/stderr
        **kwargs: Additional PartitionerConfig parameters (balance, boundary_factor,
                 num_optimizing_cuts, small_component_size, max_cell_sizes)
    
    Returns:
        Dict with success, duration, stdout, stderr, and optionally error keys
    
    Example:
        >>> import osrm
        >>> 
        >>> # First extract
        >>> osrm.extract('data.osm.pbf', profile_path='profiles/car.lua')
        >>> 
        >>> # Then partition for MLD
        >>> result = osrm.partition('data.osrm')
        >>> 
        >>> # Finally customize
        >>> osrm.customize('data.osrm')
    """
    config = osrm_ext.PartitionerConfig()
    config.UseDefaultOutputNames(Path(input_path))
    
    if threads is not None:
        config.requested_num_threads = threads
    else:
        # Default to CPU count to avoid TBB issues with 0 threads
        import os
        config.requested_num_threads = os.cpu_count() or 1
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    if capture_output or progress_callback is not None:
        return osrm_ext.partition_with_capture(config, verbosity, progress_callback)
    else:
        osrm_ext.partition(config, verbosity)
        return {"success": True}


def customize(
    input_path: str,
    threads: Optional[int] = None,
    verbosity: str = "INFO",
    progress_callback: Optional[Callable[[str], None]] = None,
    capture_output: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Customize partitioned graph for MLD (Multi-Level Dijkstra) algorithm.
    
    This is the final step for MLD routing (after extract and partition).
    
    Args:
        input_path: Base path to .osrm files (output from partition)
        threads: Number of threads to use. If None, uses all available CPU cores.
        verbosity: Log level - one of: "NONE", "ERROR", "WARNING", "INFO", "DEBUG"
        progress_callback: Optional callback function(line: str) for progress updates
        capture_output: If True, capture and return stdout/stderr
        **kwargs: Additional CustomizationConfig parameters
    
    Returns:
        Dict with success, duration, stdout, stderr, and optionally error keys
    
    Example:
        >>> import osrm
        >>> 
        >>> # Complete MLD pipeline
        >>> osrm.extract('data.osm.pbf', profile_path='profiles/car.lua')
        >>> osrm.partition('data.osrm')
        >>> result = osrm.customize('data.osrm')
        >>> 
        >>> # Now ready for routing with MLD
        >>> engine = osrm.OSRM('data.osrm', algorithm='MLD')
    """
    config = osrm_ext.CustomizationConfig()
    config.UseDefaultOutputNames(Path(input_path))
    
    if threads is not None:
        config.requested_num_threads = threads
    else:
        # Default to CPU count to avoid TBB issues with 0 threads
        import os
        config.requested_num_threads = os.cpu_count() or 1
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    if capture_output or progress_callback is not None:
        return osrm_ext.customize_with_capture(config, verbosity, progress_callback)
    else:
        osrm_ext.customize(config, verbosity)
        return {"success": True}
