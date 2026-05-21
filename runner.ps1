$repo = 'C:\Users\MiloBuwalda\Documents\GitHub\or_py'
Set-Location $repo
$env:PYTHONPATH       = (Join-Path $repo 'src')
$env:PYTHONIOENCODING = 'utf-8'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$env:ANTHROPIC_API_KEY = & $py -c "from dotenv import dotenv_values; print(dotenv_values(r'$repo\.env')['ANTHROPIC_API_KEY'])"

# >>> EDIT THESE FOUR LINES TO POINT AT YOUR NEW IMAGES <<<
$ref1 = 'C:\Falcker\cloud\falcker\AI\Operator Round TP6\testsets\stream_test_set_04_water_test\1.jpg'
$ref2 = 'C:\Falcker\cloud\falcker\AI\Operator Round TP6\testsets\stream_test_set_04_water_test\2.jpg'
$ex   = 'C:\Falcker\cloud\falcker\AI\Operator Round TP6\testsets\references\water_ref_02.png:water_pooling'
$tgt  = 'C:\Falcker\cloud\falcker\AI\Operator Round TP6\testsets\stream_test_set_04_water_test\3.jpg'

$outDir = Join-Path $repo 'dev\prompt_sweep'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$prompts = Get-ChildItem .\src\change_detection\prompts\*.txt | Sort-Object Name
foreach ($p in $prompts) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($p.Name)
    $out  = Join-Path $outDir "$stem.jpg"
    Write-Host ""
    Write-Host "===== $($p.Name) =====" -ForegroundColor Cyan
    & $py .\src\change_detection\claude_change_detect.py `
        --ref $ref1 --ref $ref2 `
        --example $ex `
        $tgt `
        --prompt $p.FullName `
        --label $stem `
        --output $out
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($p.Name) (exit $LASTEXITCODE)" -ForegroundColor Red }
}

Write-Host ""
Write-Host "===== runs/ (latest) ====="
Get-ChildItem (Join-Path $repo 'runs') | Sort-Object LastWriteTime -Descending | Select-Object -First 10 | Format-Table Name, LastWriteTime -AutoSize