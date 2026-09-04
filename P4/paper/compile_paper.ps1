$ErrorActionPreference = "Stop"

$paperRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $paperRoot "..\..")).Path
$bundledTectonicCandidates = @(
    (Join-Path $repoRoot "P1\tools\tectonic\tectonic-0.17.0\tectonic.exe"),
    (Join-Path $repoRoot "P1\tools\tectonic\tectonic-0.16.9\tectonic.exe")
)
$bundledTectonic = $bundledTectonicCandidates |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
$tectonic = $env:TECTONIC_EXE
if ([string]::IsNullOrWhiteSpace($tectonic) -and (Test-Path -LiteralPath $bundledTectonic)) {
    $tectonic = $bundledTectonic
}
if ([string]::IsNullOrWhiteSpace($tectonic)) {
    $tectonicCommand = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($null -ne $tectonicCommand) { $tectonic = $tectonicCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($tectonic) -or -not (Test-Path -LiteralPath $tectonic)) {
    throw "Tectonic was not found. Install it, bundle it under P1/tools/tectonic, or set TECTONIC_EXE."
}

# Keep downloaded bundles outside version control while avoiding user-profile
# cache corruption on Windows. A clean checkout populates this cache on first use.
$projectCache = Join-Path $repoRoot "P4\.tectonic-cache"
New-Item -ItemType Directory -Path $projectCache -Force | Out-Null
$projectCache = (Resolve-Path -LiteralPath $projectCache).Path
$env:TECTONIC_CACHE_DIR = $projectCache
$tectonicArgs = @()

$workRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("p4_manuscript_" + [guid]::NewGuid().ToString("N"))
$enOut = Join-Path $workRoot "en"
$zhOut = Join-Path $workRoot "zh"
$suppOut = Join-Path $workRoot "supp"
New-Item -ItemType Directory -Path $enOut, $zhOut, $suppOut -Force | Out-Null

try {
    Push-Location $paperRoot
    try {
        & $tectonic "dfsc_primitive_protocol_en.tex" --outdir $enOut @tectonicArgs
        if ($LASTEXITCODE -ne 0) { throw "English compilation failed with exit code $LASTEXITCODE" }

        & $tectonic "dfsc_primitive_protocol_zh.tex" --outdir $zhOut @tectonicArgs
        if ($LASTEXITCODE -ne 0) { throw "Chinese compilation failed with exit code $LASTEXITCODE" }

        & $tectonic "dfsc_primitive_protocol_supplement.tex" --outdir $suppOut @tectonicArgs
        if ($LASTEXITCODE -ne 0) { throw "Supplement compilation failed with exit code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }

    Copy-Item -LiteralPath (Join-Path $enOut "dfsc_primitive_protocol_en.pdf") `
        -Destination (Join-Path $paperRoot "dfsc_primitive_protocol_en.pdf") -Force
    Copy-Item -LiteralPath (Join-Path $zhOut "dfsc_primitive_protocol_zh.pdf") `
        -Destination (Join-Path $paperRoot "dfsc_primitive_protocol_zh.pdf") -Force
    Copy-Item -LiteralPath (Join-Path $suppOut "dfsc_primitive_protocol_supplement.pdf") `
        -Destination (Join-Path $paperRoot "dfsc_primitive_protocol_supplement.pdf") -Force

    Write-Output "Compiled the current journal-neutral main and supplementary PDFs successfully."
}
finally {
    if (Test-Path -LiteralPath $workRoot) {
        Remove-Item -LiteralPath $workRoot -Recurse -Force
    }
}
