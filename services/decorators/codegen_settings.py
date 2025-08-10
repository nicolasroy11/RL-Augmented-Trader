PROJECT_ROOT = '/Users/nroy/Desktop/veritone-energy/venergy/iso_service_app'

# path and port the API will be served over
API_URL = "http://localhost"
API_PORT = 8000

# where you want the codegen output files to appear. At the moment, these directories should all be the same.
DTOS_OUTDIR = "/Users/nroy/Desktop/veritone-energy/ui/web-energy/src/generated"
ENDPOINTS_OUTDIR = "/Users/nroy/Desktop/veritone-energy/ui/web-energy/src/generated"
ENUMS_OUTDIR = "/Users/nroy/Desktop/veritone-energy/ui/web-energy/src/generated"

# these are the driectories inside the PROJECT_ROOT you want codegen to skip (relative paths only)
EXCLUDED_DIRECTORIES = [
    'codegen',
    'tests',
    'migrations'
]

# these are directories outside PROJECT_ROOT you want to include (absolute paths only)
INCLUDED_DIRECTORIES = [
    '/Users/nroy/Desktop/veritone-energy/venergy/observables'
]