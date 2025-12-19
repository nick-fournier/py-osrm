"""Pytest configuration and fixtures for py-osrm tests.

This conftest.py provides automatic test infrastructure:

1. **Automatic osrm-datastore startup**: The osrm_datastore fixture automatically
   starts osrm-datastore with test data before any tests run. This enables tests
   that use osrm.OSRM() without arguments (shared memory mode) to work without
   manual setup.

2. **Smart detection**: If shared memory is already available, it won't start
   a new datastore process.

3. **Automatic cleanup**: The datastore is terminated after all tests complete.

To run tests:
    pytest tests/                    # Datastore starts automatically
    pytest tests/test_route.py       # Works for any test file
    
To disable automatic datastore:
    pytest tests/ --no-cov           # (no special flag needed, it's automatic)
"""
import subprocess
import time
import pytest
import osrm
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def osrm_datastore():
    """Start osrm-datastore with test data for shared memory tests.
    
    This fixture automatically starts before any tests run and stops after all tests complete.
    It enables tests that use osrm.OSRM() without arguments to work.
    """
    # Path to test data
    data_path = Path(__file__).parent / "data" / "ch" / "monaco.osrm"
    
    # Check if datastore is needed by trying to create OSRM instance
    try:
        test_osrm = osrm.OSRM()
        # Already have shared memory available
        print("\n✓ Shared memory already available")
        yield
        return
    except RuntimeError:
        # Need to start datastore
        pass
    
    # Start osrm-datastore in background
    print(f"\n→ Starting osrm-datastore with {data_path}")
    try:
        process = subprocess.Popen(
            ["python3", "-m", "osrm", "datastore", str(data_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for datastore to initialize and load data
        time.sleep(1.5)
        
        # Verify shared memory is accessible
        try:
            test_osrm = osrm.OSRM()
            print("✓ osrm-datastore started successfully")
        except RuntimeError as e:
            # Get output if process failed
            if process.poll() is not None:
                output, _ = process.communicate()
                raise RuntimeError(f"osrm-datastore failed to start:\n{output}")
            process.terminate()
            raise RuntimeError(f"osrm-datastore started but shared memory not accessible: {e}")
        
        yield
        
        # Cleanup: terminate datastore
        print("\n→ Stopping osrm-datastore")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("✓ osrm-datastore stopped")
        
    except FileNotFoundError:
        pytest.skip("osrm datastore command not available")
    except Exception as e:
        print(f"⚠ Could not start osrm-datastore: {e}")
        print("  Shared memory tests will be skipped")
        yield
