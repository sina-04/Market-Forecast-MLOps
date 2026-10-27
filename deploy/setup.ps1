$ErrorActionPreference = 'Stop'
python "$PSScriptRoot/setup.py"
if ($LASTEXITCODE -ne 0) { throw 'Secret initialization failed.' }
