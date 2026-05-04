"""Bulk processing functions for py-osrm using concurrent execution."""

import asyncio
import math
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, overload

import aiohttp

from ._params import RouteParameters as _RouteParameters, set_param as _set_param

# Default concurrent HTTP requests — high enough for throughput,
# low enough to avoid overwhelming a remote server
_DEFAULT_HTTP_WORKERS = 16

# TypeVar for DataFrame type (Polars DataFrame)
DataFrameT = TypeVar('DataFrameT')


def _bulk_route_http(
    osrm_instance,
    rows: List[Dict],
    build_params_fn: Callable,
    concurrency: int,
    fail_fast: bool,
    show_progress: bool = True,
) -> List[Optional[Dict]]:
    """Async HTTP routing with aiohttp and semaphore-based concurrency control."""

    async def _run():
        sem = asyncio.Semaphore(concurrency)
        connector = aiohttp.TCPConnector(limit=concurrency, keepalive_timeout=30)

        async def _fetch_one(session: aiohttp.ClientSession, idx: int, row: Dict) -> tuple:
            kw = build_params_fn(row)
            coordinates = kw.pop('coordinates')
            request_data = osrm_instance._build_request('route', coordinates, kw)

            async with sem:
                try:
                    async with session.get(
                        request_data['url'],
                        params=request_data['params'],
                        timeout=aiohttp.ClientTimeout(total=osrm_instance.timeout),
                    ) as resp:
                        if resp.status == 200:
                            return idx, await resp.json()
                        return idx, None
                except Exception:
                    if fail_fast:
                        raise
                    return idx, None

        # Set up optional progress bar
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(rows), desc="Routing (HTTP)", unit="req")
            except ImportError:
                pass

        results = [None] * len(rows)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [asyncio.ensure_future(_fetch_one(session, i, row))
                     for i, row in enumerate(rows)]
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results[idx] = result
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()
        return results

    # Run the event loop — handle case where one is already running (e.g. Jupyter)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context (Jupyter, etc.) — use thread to run new loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run, _run())
            return future.result()
    else:
        return asyncio.run(_run())



@overload
def bulk_route(
    osrm_instance,
    df: Dict[str, List],
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> Dict[str, List]: ...


@overload
def bulk_route(
    osrm_instance,
    df: DataFrameT,
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> DataFrameT: ...


def bulk_route(
    osrm_instance,
    df: Union[DataFrameT, Dict[str, List]],
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> Union[DataFrameT, Dict[str, List]]:
    """
    Process multiple route requests in parallel from a DataFrame.
    
    Args:
        osrm_instance: An initialized OSRM instance
        df: Polars DataFrame or dict-of-lists with columns:
            Required: origin_lon, origin_lat, dest_lon, dest_lat
            Optional per-row params: steps, alternatives, annotations, geometries, 
                                     overview, radiuses, bearings, exclude
            For multi-waypoint routes: waypoints (list of (lon, lat) tuples)
        max_workers: Number of parallel workers. For native OSRM, this is ignored
            (TBB handles threading). For HTTP clients, defaults to 8 to avoid
            overwhelming the remote server. Override with caution.
        fail_fast: If True, raise on first error; if False, collect all results
        timeout: Timeout in seconds for individual route requests
        show_progress: If True (default), display progress bar with processing rate and error count
        **default_params: Default parameters applied to all routes (overridden by row params)
    
    Returns:
        Polars DataFrame (or dict-of-lists) with original columns plus:
            - distance: Route distance in meters
            - duration: Route duration in seconds
            - geometry: Route geometry (format depends on geometries param)
            - success: Boolean indicating if route succeeded
            - error: Error message if success=False, None otherwise
    
    Examples:
        >>> import polars as pl
        >>> import osrm
        >>> 
        >>> osrm_instance = osrm.OSRM("path/to/data.osrm")
        >>> 
        >>> # Create DataFrame with OD pairs
        >>> df = pl.DataFrame({
        ...     "origin_lon": [7.41337, 7.41862],
        ...     "origin_lat": [43.72956, 43.73216],
        ...     "dest_lon": [7.41546, 7.42000],
        ...     "dest_lat": [43.73077, 43.73300]
        ... })
        >>> 
        >>> # Process all routes in parallel
        >>> results = osrm.bulk_route(osrm_instance, df, steps=True, geometries="geojson")
        >>> print(results.select(["distance", "duration", "success"]))
    """
    # Determine if input is Polars DataFrame or dict-of-lists
    try:
        import polars as pl
        is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars = False
    
    # Convert to list of dicts for processing
    if is_polars:
        rows = df.to_dicts()
    elif isinstance(df, dict):
        # Convert dict-of-lists to list-of-dicts
        keys = list(df.keys())
        rows = [dict(zip(keys, values)) for values in zip(*df.values())]
    else:
        raise TypeError("df must be a Polars DataFrame or dict-of-lists")
    
    # Validate required columns
    required_cols = ['origin_lon', 'origin_lat', 'dest_lon', 'dest_lat']
    first_row = rows[0] if rows else {}
    missing_cols = [col for col in required_cols if col not in first_row]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # Optional progress bar
    progress_bar = None
    error_count = 0
    if show_progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=len(rows),
                desc="Routing",
                unit="req",
                unit_scale=False,
                postfix={"errors": 0}
            )
        except ImportError:
            pass  # tqdm not installed, skip progress bar
    
    results = []
    
    # Create parameter dict for each row
    def build_params_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Build RouteParameters kwargs from row data and defaults."""
        params = default_params.copy()
        
        # Extract coordinates
        if 'waypoints' in row and row['waypoints']:
            # Multi-waypoint route
            params['coordinates'] = row['waypoints']
        else:
            # Simple OD pair
            params['coordinates'] = [
                (row['origin_lon'], row['origin_lat']),
                (row['dest_lon'], row['dest_lat'])
            ]
        
        # Override with row-specific parameters
        param_cols = ['steps', 'alternatives', 'number_of_alternatives', 'annotations', 
                      'geometries', 'overview', 'continue_straight', 'radiuses', 
                      'bearings', 'exclude', 'generate_hints', 'snapping', 'approaches']
        
        for col in param_cols:
            if col in row and row[col] is not None:
                params[col] = row[col]
        
        return params

    # Build C++ RouteParameters objects for BatchRoute
    params_list = []
    for row in rows:
        kw = build_params_for_row(row)
        rp = _RouteParameters()
        coords = kw.pop('coordinates')
        rp.coordinates = coords
        for key, value in kw.items():
            _set_param(rp, key, value)
        params_list.append(rp)

    # Detect whether this is a native C++ engine (has BatchRoute) or HTTP client
    _has_batch = hasattr(osrm_instance, '_engine') and hasattr(osrm_instance._engine, 'BatchRoute')

    if _has_batch:
        # Native C++ path — TBB parallel via single GIL release
        try:
            batch_results = osrm_instance._engine.BatchRoute(params_list)
        except Exception as e:
            if fail_fast:
                raise
            batch_results = [None] * len(rows)
    else:
        # HTTP/fallback path — use async I/O for maximum throughput
        concurrency = max_workers or _DEFAULT_HTTP_WORKERS

        batch_results = _bulk_route_http(osrm_instance, rows, build_params_for_row,
                                         concurrency, fail_fast, show_progress)

    # Unpack results into row dicts
    results: List[Dict[str, Any]] = [None] * len(rows)  # type: ignore
    for i, (row, raw) in enumerate(zip(rows, batch_results)):
        result = row.copy()
        try:
            if raw is not None:
                response = raw.to_dict() if hasattr(raw, 'to_dict') else raw
                if response and 'routes' in response and len(response['routes']) > 0:
                    route = response['routes'][0]
                    result['distance'] = route.get('distance')
                    result['duration'] = route.get('duration')
                    result['geometry'] = route.get('geometry')
                    result['success'] = True
                    result['error'] = None
                else:
                    result['distance'] = None
                    result['duration'] = None
                    result['geometry'] = None
                    result['success'] = False
                    result['error'] = "No routes found"
                    if fail_fast:
                        raise RuntimeError(f"Route {i}: No routes found")
            else:
                result['distance'] = None
                result['duration'] = None
                result['geometry'] = None
                result['success'] = False
                result['error'] = "Route failed"
                if fail_fast:
                    raise RuntimeError(f"Route {i}: Route failed")
        except Exception as e:
            result['distance'] = None
            result['duration'] = None
            result['geometry'] = None
            result['success'] = False
            result['error'] = str(e)
            if fail_fast:
                raise

        if not result.get('success', False):
            error_count += 1
        if progress_bar:
            progress_bar.set_postfix({"errors": error_count})
            progress_bar.update(1)

        results[i] = result

    if progress_bar:
        progress_bar.close()
    
    # Convert results back to DataFrame or dict-of-lists
    if is_polars:
        import polars as pl
        return pl.DataFrame(results, infer_schema_length=None)
    else:
        # Convert list-of-dicts back to dict-of-lists
        if not results:
            return {}
        keys = results[0].keys()
        return {key: [r[key] for r in results] for key in keys}


@overload
def bulk_nearest(
    osrm_instance,
    df: Dict[str, List],
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> Dict[str, List]: ...


@overload
def bulk_nearest(
    osrm_instance,
    df: DataFrameT,
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> DataFrameT: ...


def bulk_nearest(
    osrm_instance,
    df: Union[DataFrameT, Dict[str, List]],
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> Union[DataFrameT, Dict[str, List]]:
    """
    Process multiple nearest requests in parallel from a DataFrame.
    
    Args:
        osrm_instance: An initialized OSRM instance
        df: Polars DataFrame or dict-of-lists with columns:
            Required: lon, lat
            Optional per-row params: number, radiuses, bearings, approaches, 
                                     exclude, generate_hints, snapping
        max_workers: Number of parallel workers (default: os.cpu_count())
        fail_fast: If True, raise on first error; if False, collect all results
        timeout: Timeout in seconds for individual nearest requests
        show_progress: If True (default), display progress bar with processing rate and error count
        **default_params: Default parameters applied to all nearest requests (overridden by row params)
    
    Returns:
        Polars DataFrame (or dict-of-lists) with original columns plus:
            - waypoint_lon: Longitude of nearest waypoint
            - waypoint_lat: Latitude of nearest waypoint
            - waypoint_name: Name of street/location
            - distance: Distance to nearest waypoint in meters
            - success: Boolean indicating if nearest succeeded
            - error: Error message if success=False, None otherwise
    
    Examples:
        >>> import polars as pl
        >>> import osrm
        >>> 
        >>> osrm_instance = osrm.OSRM("path/to/data.osrm")
        >>> 
        >>> # Create DataFrame with coordinates
        >>> df = pl.DataFrame({
        ...     "lon": [7.41337, 7.41862],
        ...     "lat": [43.72956, 43.73216]
        ... })
        >>> 
        >>> # Process all nearest requests in parallel
        >>> results = osrm.bulk_nearest(osrm_instance, df, number=3)
        >>> print(results.select(["waypoint_lon", "waypoint_lat", "distance", "success"]))
    """
    # Determine if input is Polars DataFrame or dict-of-lists
    try:
        import polars as pl
        is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars = False
    
    # Convert to list of dicts for processing
    if is_polars:
        rows = df.to_dicts()
    elif isinstance(df, dict):
        keys = list(df.keys())
        rows = [dict(zip(keys, values)) for values in zip(*df.values())]
    else:
        raise TypeError("df must be a Polars DataFrame or dict-of-lists")
    
    # Validate required columns
    required_cols = ['lon', 'lat']
    first_row = rows[0] if rows else {}
    missing_cols = [col for col in required_cols if col not in first_row]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # Optional progress bar
    progress_bar = None
    error_count = 0
    if show_progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=len(rows),
                desc="Nearest",
                unit="req",
                unit_scale=False,
                postfix={"errors": 0}
            )
        except ImportError:
            pass
    
    results = []
    
    def build_params_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Build NearestParameters kwargs from row data and defaults."""
        params = default_params.copy()
        
        # Extract coordinate
        params['coordinates'] = [(row['lon'], row['lat'])]
        
        # Override with row-specific parameters
        param_cols = ['number', 'radiuses', 'bearings', 'approaches', 
                      'exclude', 'generate_hints', 'snapping']
        
        for col in param_cols:
            if col in row and row[col] is not None:
                params[col] = row[col]
        
        return params
    
    def process_single_nearest(row: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single nearest request."""
        result = row.copy()
        
        try:
            params = build_params_for_row(row)
            response = osrm_instance.Nearest(**params)
            
            # Extract key metrics from response
            if response and 'waypoints' in response and len(response['waypoints']) > 0:
                waypoint = response['waypoints'][0]
                result['waypoint_lon'] = waypoint.get('location', [None, None])[0]
                result['waypoint_lat'] = waypoint.get('location', [None, None])[1]
                result['waypoint_name'] = waypoint.get('name')
                result['distance'] = waypoint.get('distance', 0.0)
                result['success'] = True
                result['error'] = None
            else:
                result['waypoint_lon'] = None
                result['waypoint_lat'] = None
                result['waypoint_name'] = None
                result['distance'] = None
                result['success'] = False
                result['error'] = "No waypoints found"
                
        except Exception as e:
            result['waypoint_lon'] = None
            result['waypoint_lat'] = None
            result['waypoint_name'] = None
            result['distance'] = None
            result['success'] = False
            result['error'] = str(e)
            
            if fail_fast:
                raise
        
        return result
    
    def process_chunk(chunk: List[tuple]) -> List[tuple]:
        """Process a chunk of (index, row) pairs sequentially within one thread."""
        return [(index, process_single_nearest(row, index)) for index, row in chunk]

    # Split rows into chunks (one per worker) to minimize task submission overhead
    chunk_size = max(1, math.ceil(len(rows) / max_workers))
    indexed_rows = list(enumerate(rows))
    chunks = [indexed_rows[i:i + chunk_size] for i in range(0, len(indexed_rows), chunk_size)]

    # Process nearest request chunks in parallel
    results: List[Dict[str, Any]] = [None] * len(rows)  # type: ignore

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk): chunk
            for chunk in chunks
        }

        try:
            for future in as_completed(future_to_chunk, timeout=timeout):
                try:
                    chunk_results = future.result()
                    for index, result in chunk_results:
                        results[index] = result
                        if not result.get('success', False):
                            error_count += 1
                        if progress_bar:
                            progress_bar.set_postfix({"errors": error_count})
                            progress_bar.update(1)
                except Exception as e:
                    if fail_fast:
                        raise
                    failed_chunk = future_to_chunk[future]
                    for index, row in failed_chunk:
                        error_count += 1
                        results[index] = row.copy()
                        results[index].update({
                            'waypoint_lon': None,
                            'waypoint_lat': None,
                            'waypoint_name': None,
                            'distance': None,
                            'success': False,
                            'error': str(e)
                        })
                        if progress_bar:
                            progress_bar.set_postfix({"errors": error_count})
                            progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()
    
    # Convert results back to DataFrame or dict-of-lists
    if is_polars:
        import polars as pl
        return pl.DataFrame(results, infer_schema_length=None)
    else:
        if not results:
            return {}
        keys = results[0].keys()
        return {key: [r[key] for r in results] for key in keys}


@overload
def bulk_match(
    osrm_instance,
    df: Dict[str, List],
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> Dict[str, List]: ...


@overload
def bulk_match(
    osrm_instance,
    df: DataFrameT,
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> DataFrameT: ...


def bulk_match(
    osrm_instance,
    df: Union[DataFrameT, Dict[str, List]],
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
    timeout: Optional[float] = None,
    show_progress: bool = True,
    **default_params
) -> Union[DataFrameT, Dict[str, List]]:
    """
    Process multiple match requests in parallel from a DataFrame.
    
    Args:
        osrm_instance: An initialized OSRM instance
        df: Polars DataFrame or dict-of-lists with columns:
            Required: coordinates (list of (lon, lat) tuples for GPS trace)
            Optional per-row params: timestamps, steps, geometries, annotations,
                                     overview, radiuses, bearings, approaches,
                                     exclude, gaps, tidy, waypoints
        max_workers: Number of parallel workers (default: os.cpu_count())
        fail_fast: If True, raise on first error; if False, collect all results
        timeout: Timeout in seconds for individual match requests
        show_progress: If True (default), display progress bar with processing rate and error count
        **default_params: Default parameters applied to all matches (overridden by row params)
    
    Returns:
        Polars DataFrame (or dict-of-lists) with original columns plus:
            - distance: Matched route distance in meters
            - duration: Matched route duration in seconds
            - confidence: Matching confidence score
            - geometry: Route geometry (format depends on geometries param)
            - success: Boolean indicating if match succeeded
            - error: Error message if success=False, None otherwise
    
    Examples:
        >>> import polars as pl
        >>> import osrm
        >>> 
        >>> osrm_instance = osrm.OSRM("path/to/data.osrm")
        >>> 
        >>> # Create DataFrame with GPS traces
        >>> df = pl.DataFrame({
        ...     "coordinates": [
        ...         [(7.41337, 43.72956), (7.41546, 43.73077), (7.41862, 43.73216)],
        ...         [(7.42000, 43.73300), (7.42150, 43.73400), (7.42300, 43.73500)]
        ...     ]
        ... })
        >>> 
        >>> # Process all match requests in parallel
        >>> results = osrm.bulk_match(osrm_instance, df, geometries="geojson")
        >>> print(results.select(["distance", "duration", "confidence", "success"]))
    """
    # Determine if input is Polars DataFrame or dict-of-lists
    try:
        import polars as pl
        is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars = False
    
    # Convert to list of dicts for processing
    if is_polars:
        rows = df.to_dicts()
    elif isinstance(df, dict):
        keys = list(df.keys())
        rows = [dict(zip(keys, values)) for values in zip(*df.values())]
    else:
        raise TypeError("df must be a Polars DataFrame or dict-of-lists")
    
    # Validate required columns
    required_cols = ['coordinates']
    first_row = rows[0] if rows else {}
    missing_cols = [col for col in required_cols if col not in first_row]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # Optional progress bar
    progress_bar = None
    error_count = 0
    if show_progress:
        try:
            from tqdm import tqdm
            progress_bar = tqdm(
                total=len(rows),
                desc="Matching",
                unit="req",
                unit_scale=False,
                postfix={"errors": 0}
            )
        except ImportError:
            pass
    
    results = []
    
    def build_params_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Build MatchParameters kwargs from row data and defaults."""
        params = default_params.copy()
        
        # Extract coordinates
        params['coordinates'] = row['coordinates']
        
        # Override with row-specific parameters
        param_cols = ['timestamps', 'steps', 'geometries', 'annotations', 
                      'overview', 'radiuses', 'bearings', 'approaches',
                      'exclude', 'gaps', 'tidy', 'waypoints', 'generate_hints', 'snapping']
        
        for col in param_cols:
            if col in row and row[col] is not None:
                params[col] = row[col]
        
        return params
    
    def process_single_match(row: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single match request."""
        result = row.copy()
        
        try:
            params = build_params_for_row(row)
            response = osrm_instance.Match(**params)
            
            # Extract key metrics from response
            if response and 'matchings' in response and len(response['matchings']) > 0:
                matching = response['matchings'][0]
                result['distance'] = matching.get('distance')
                result['duration'] = matching.get('duration')
                result['confidence'] = matching.get('confidence')
                result['geometry'] = matching.get('geometry')
                result['success'] = True
                result['error'] = None
            else:
                result['distance'] = None
                result['duration'] = None
                result['confidence'] = None
                result['geometry'] = None
                result['success'] = False
                result['error'] = "No matchings found"
                
        except Exception as e:
            result['distance'] = None
            result['duration'] = None
            result['confidence'] = None
            result['geometry'] = None
            result['success'] = False
            result['error'] = str(e)
            
            if fail_fast:
                raise
        
        return result
    
    def process_chunk(chunk: List[tuple]) -> List[tuple]:
        """Process a chunk of (index, row) pairs sequentially within one thread."""
        return [(index, process_single_match(row, index)) for index, row in chunk]

    # Split rows into chunks (one per worker) to minimize task submission overhead
    chunk_size = max(1, math.ceil(len(rows) / max_workers))
    indexed_rows = list(enumerate(rows))
    chunks = [indexed_rows[i:i + chunk_size] for i in range(0, len(indexed_rows), chunk_size)]

    # Process match request chunks in parallel
    results: List[Dict[str, Any]] = [None] * len(rows)  # type: ignore

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk): chunk
            for chunk in chunks
        }

        try:
            for future in as_completed(future_to_chunk, timeout=timeout):
                try:
                    chunk_results = future.result()
                    for index, result in chunk_results:
                        results[index] = result
                        if not result.get('success', False):
                            error_count += 1
                        if progress_bar:
                            progress_bar.set_postfix({"errors": error_count})
                            progress_bar.update(1)
                except Exception as e:
                    if fail_fast:
                        raise
                    failed_chunk = future_to_chunk[future]
                    for index, row in failed_chunk:
                        error_count += 1
                        results[index] = row.copy()
                        results[index].update({
                            'distance': None,
                            'duration': None,
                            'confidence': None,
                            'geometry': None,
                            'success': False,
                            'error': str(e)
                        })
                        if progress_bar:
                            progress_bar.set_postfix({"errors": error_count})
                            progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()
    
    # Convert results back to DataFrame or dict-of-lists
    if is_polars:
        import polars as pl
        return pl.DataFrame(results, infer_schema_length=None)
    else:
        if not results:
            return {}
        keys = results[0].keys()
        return {key: [r[key] for r in results] for key in keys}



