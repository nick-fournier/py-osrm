"""Command-line interface for OSRM preprocessing tools."""
import os
import site
import subprocess
import sys


HELP_TEXT = """
py-osrm - Python bindings for OSRM routing engine

USAGE:
    python -m osrm <command> [options]

PREPROCESSING COMMANDS:
    extract     Extract road network from OSM data
    partition   Partition graph for MLD algorithm
    customize   Customize graph for specific profile
    contract    Contract graph for CH algorithm
    components  Find connected components
    datastore   Load/manage shared memory datastore
    routed      Start OSRM HTTP server

EXAMPLES:
    # Extract road network
    python -m osrm extract -p profiles/car.lua map.osm.pbf
    
    # Contract for CH algorithm
    python -m osrm contract map.osrm
    
    # Load into shared memory
    python -m osrm datastore map.osrm

PYTHON API:
    import osrm
    
    # Native (local files)
    engine = osrm.OSRM("map.osrm")
    result = engine.Route([(lon1, lat1), (lon2, lat2)])
    
    # HTTP (remote server)
    client = osrm.OSRM_HTTP("http://localhost:5000")
    result = client.Route([(lon1, lat1), (lon2, lat2)])
    
    # Bulk processing
    import polars as pl
    df = pl.DataFrame({"origin_lon": [...], "origin_lat": [...], 
                       "dest_lon": [...], "dest_lat": [...]})
    results = osrm.bulk_route(engine, df)

For more information: https://github.com/gis-ops/py-osrm
"""

if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
    print(HELP_TEXT)
    sys.exit(0)

searchpaths = site.getsitepackages()
if(site.ENABLE_USER_SITE):
    searchpaths.append(site.getusersitepackages())

exec = ""

for path in searchpaths:
    currpath = path + "/bin/"
    if os.path.isfile(currpath + "osrm-datastore") or os.path.isfile(currpath + "osrm-datastore.exe"):
        exec = currpath
        break

if not exec:
    print("Python OSRM executables not found")
    sys.exit(1)

if(sys.argv[1] == "components"):
    exec += "osrm-components"

elif(sys.argv[1] == "contract"):
    exec += "osrm-contract"

elif(sys.argv[1] == "customize"):
    exec += "osrm-customize"

elif(sys.argv[1] == "datastore"):
    exec += "osrm-datastore"

elif(sys.argv[1] == "extract"):
    exec += "osrm-extract"

elif(sys.argv[1] == "partition"):
    exec += "osrm-partition"

elif(sys.argv[1] == "routed"):
    exec += "osrm-routed"

for i in range(2, len(sys.argv)):
    exec += " " + sys.argv[i]

# will stream any output to the shell
# shell=True is safe here and lets people use "~" in paths etc
proc = subprocess.run(exec, encoding="utf-8", shell=True)
sys.exit(proc.returncode)
