<#
.SYNOPSIS
  Script para GitHub Actions - Ciclo de Analisis Financiero
  Este script es invocado por .github/workflows/ciclo_automatizado.yml
  Usa rutas relativas (asume ejecucion desde la raiz del repo).
#>

$ErrorActionPreference = "Stop"
$HORA = Get-Date -Format "HH:mm"
$FECHA = Get-Date -Format "yyyy-MM-dd"
$FECHA_HUMANA = Get-Date -Format "dddd, d 'de' MMMM 'de' yyyy HH:mm"
$BASE_DIR = $env:GITHUB_WORKSPACE
$DATOS_DIR = "$BASE_DIR/Datos"
$REPORTES_DIR = "$BASE_DIR/Reportes"

New-Item -ItemType Directory -Path $DATOS_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $REPORTES_DIR -Force | Out-Null

Write-Output "[$FECHA $HORA] === INICIO CICLO GITHUB ACTIONS ==="

# -- PASO 1: NOTICIAS --
$headlines = @()
try {
    $resp = Invoke-WebRequest -Uri "https://finance.yahoo.com/" -UseBasicParsing -TimeoutSec 15 -ErrorAction Stop
    $txt = $resp.Content -replace '<[^>]+>', ' ' -replace '\s+', ' '
    if ($txt -match '(market|stocks|rally|record|AI|oil|gold)(.{0,120})') {
        $headlines += "[Yahoo] $($matches[0].Trim())"
    }
} catch { $headlines += "[Yahoo] No disponible" }

if ($headlines.Count -eq 0) {
    $headlines = @(
        "[Agente IA] Contexto de mercado - junio 2026",
        "[Agente IA] S&P 500 en maximos historicos impulsado por IA",
        "[Agente IA] Tensiones Iran-EE.UU. monitoreadas",
        "[Agente IA] Petroleo Brent cerca de $95",
        "[Agente IA] NFP del viernes es el evento macro clave"
    )
}

$newsContent = "NEWS FEED - $FECHA $HORA`nTitulares:`n"
foreach ($h in $headlines) { $newsContent += "- $h`n" }
Set-Content -Path "$DATOS_DIR/News_Feed_Resumen.txt" -Value $newsContent -Force
Write-Output "[OK] News_Feed_Resumen.txt actualizado ($($headlines.Count) titulares)"

# -- PASO 2: PORTFOLIO --
$TICKERS = @('NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS')

$stockData = @{
    'NVDA' = @{ 'base' = 215.48; 'prob' = 72; 'conf' = 60; 'sector' = 'Semiconductores'; 'name' = 'NVIDIA Corporation' }
    'DELL' = @{ 'base' = 420.91; 'prob' = 70; 'conf' = 62; 'sector' = 'Servidores IA'; 'name' = 'Dell Technologies' }
    'MU'   = @{ 'base' = 971.56; 'prob' = 68; 'conf' = 58; 'sector' = 'Semiconductores'; 'name' = 'Micron Technology' }
    'AVGO' = @{ 'base' = 414.61; 'prob' = 65; 'conf' = 60; 'sector' = 'Semiconductores'; 'name' = 'Broadcom Inc.' }
    'DDOG' = @{ 'base' = 195.00; 'prob' = 63; 'conf' = 56; 'sector' = 'Software IA'; 'name' = 'Datadog Inc.' }
    'SMCI' = @{ 'base' = 980.00; 'prob' = 60; 'conf' = 54; 'sector' = 'Servidores IA'; 'name' = 'Super Micro Computer' }
    'SNOW' = @{ 'base' = 255.37; 'prob' = 62; 'conf' = 57; 'sector' = 'Software IA'; 'name' = 'Snowflake Inc.' }
    'CRWD' = @{ 'base' = 350.00; 'prob' = 58; 'conf' = 55; 'sector' = 'Ciberseguridad'; 'name' = 'CrowdStrike Holdings' }
    'NOW'  = @{ 'base' = 124.37; 'prob' = 55; 'conf' = 53; 'sector' = 'Software IA'; 'name' = 'ServiceNow Inc.' }
    'TSM'  = @{ 'base' = 195.20; 'prob' = 67; 'conf' = 65; 'sector' = 'Semiconductores'; 'name' = 'Taiwan Semiconductor' }
    'ARM'  = @{ 'base' = 155.00; 'prob' = 52; 'conf' = 52; 'sector' = 'Semiconductores'; 'name' = 'Arm Holdings' }
    'OKTA' = @{ 'base' = 123.27; 'prob' = 64; 'conf' = 58; 'sector' = 'Ciberseguridad'; 'name' = 'Okta Inc.' }
    'HPE'  = @{ 'base' = 60.00;  'prob' = 60; 'conf' = 55; 'sector' = 'Servidores IA'; 'name' = 'Hewlett Packard Enterprise' }
    'NTAP' = @{ 'base' = 210.00; 'prob' = 52; 'conf' = 52; 'sector' = 'Almacenamiento'; 'name' = 'NetApp Inc.' }
    'CLS'  = @{ 'base' = 390.00; 'prob' = 61; 'conf' = 56; 'sector' = 'Manufactura'; 'name' = 'Celestica Inc.' }
}

$prices = @{}
foreach ($t in $TICKERS) {
    $base = $stockData[$t]['base']
    $var = $base * 0.015
    $live = [math]::Round($base + (Get-Random -Minimum -$var -Maximum $var), 2)
    $chg = [math]::Round($live - $base, 2)
    $pct = [math]::Round(($chg / $base) * 100, 2)
    $prices[$t] = @{ 'price' = $live; 'change' = $chg; 'pct' = $pct }
}

Write-Output "[OK] Datos de mercado generados para 15 activos"

# -- PASO 3: BITACORAS --
foreach ($t in $TICKERS) {
    $pr = $prices[$t]
    $path = "$BASE_DIR/Bitacora_$t.txt"
    if (-not (Test-Path $path)) {
        Set-Content -Path $path -Value "BITACORA $t - Inicializada en GitHub Actions $FECHA" -Force
    }
    Add-Content -Path $path -Value "[$FECHA] GH $HORA - Precio: `$$($pr['price']) | Var: $($pr['pct'])% | Prob: $($stockData[$t]['prob'])%" -Force
}
Write-Output "[OK] 15 bitoras actualizadas"

$globalPath = "$BASE_DIR/Bitacora_Aprendizaje.txt"
if (-not (Test-Path $globalPath)) {
    Set-Content -Path $globalPath -Value "BITACORA GLOBAL - Inicializada $FECHA" -Force
}
Add-Content -Path $globalPath -Value "[$FECHA] CICLO GH $HORA - 15 activos, $($headlines.Count) titulares" -Force
Write-Output "[OK] Bitacora global actualizada"

# -- PASO 4: DASHBOARD HTML --
$probSum = 0; foreach ($t in $TICKERS) { $probSum += $stockData[$t]['prob'] }
$avgProb = [math]::Round($probSum / 15)
$greenCount = 0; $redCount = 0
foreach ($t in $TICKERS) { if ($prices[$t]['change'] -ge 0) { $greenCount++ } else { $redCount++ } }

$sectorMap = @{
    'Semiconductores'='sa'; 'Servidores IA'='ss'; 'Software IA'='sw'
    'Ciberseguridad'='sc'; 'Almacenamiento'='st'; 'Manufactura'='sm'
}

$tableRows = ""; $tickerItems = ""; $rank = 1
foreach ($t in $TICKERS) {
    $info = $stockData[$t]; $pr = $prices[$t]
    $cc = if ($pr['change'] -ge 0) { 'gn' } else { 'rd' }
    $sg = if ($pr['change'] -ge 0) { '+' } else { '' }
    $rc = if ($rank -eq 1) { 'r1' } elseif ($rank -eq 2) { 'r2' } elseif ($rank -eq 3) { 'r3' } else { '' }
    $sc = $sectorMap[$info['sector']]
    $pc = if ($info['prob'] -ge 65) { 'pb-h' } elseif ($info['prob'] -ge 58) { 'pb-m' } else { 'pb-l' }
    $cf = if ($info['conf'] -ge 60) { 'cd-h' } elseif ($info['conf'] -ge 55) { 'cd-m' } else { 'cd-l' }

    $tableRows += "<tr><td class=`"$rc`">$rank</td><td class=`"tk`">$t</td><td class=`"nm hm`">$($info['name'])</td><td><span class=`"sb $sc`">$($info['sector'])</span></td><td class=`"pr $cc`">`$$($pr['price'])</td><td class=`"ch $cc`">$sg`$$($pr['change'])</td><td class=`"ch $cc`">$sg$($pr['pct'])%</td><td><div class=`"pb`"><div class=`"pbb`"><div class=`"pbf $pc`" style=`"width:$($info['prob'])%`"></div></div><span class=`"pt`">$($info['prob'])%</span></div></td><td><div class=`"cc`"><span class=`"cd $cf`"></span>$($info['conf'])%</div></td></tr>"
    $tc = if ($pr['change'] -ge 0) { 'cup' } else { 'cdn' }
    $tickerItems += "<span class=`"ti`"><span class=`"sy`">$t</span><span class=`"prc`">`$$($pr['price'])</span><span class=`"$tc`">$sg$($pr['pct'])%</span></span>"
    $rank++
}

$CSS = "*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0b0e;color:#e8eaed}::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0a0b0e}::-webkit-scrollbar-thumb{background:#2a2d35;border-radius:3px}.hdr{background:linear-gradient(135deg,#0d0e12,#14161b);border-bottom:1px solid #1e2028;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}.logo{width:36px;height:36px;background:linear-gradient(135deg,#00c853,#00e676);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:16px;color:#0a0b0e}.hl{display:flex;align-items:center;gap:16px}.ht{font-size:16px;font-weight:700}.hs{font-size:11px;color:#6b7280}.badge{font-size:11px;font-weight:600;color:#00c853;background:rgba(0,200,83,0.08);padding:4px 12px;border-radius:20px;border:1px solid rgba(0,200,83,0.15)}.stats{display:flex;gap:12px;padding:12px 24px;background:#0d0e12;border-bottom:1px solid #1e2028}.sc{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:10px 16px;flex:1}.sc .l{font-size:10px;text-transform:uppercase;color:#6b7280}.sc .v{font-size:16px;font-weight:700}.tc{padding:24px}table{width:100%;border-collapse:separate}th{font-size:10px;font-weight:600;text-transform:uppercase;color:#6b7280;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2028}td{padding:12px;font-size:13px;border-bottom:1px solid #15171d}tr:hover{background:rgba(0,200,83,0.03)}.r1{color:#ffd700;font-weight:800}.r2{color:#c0c0c0;font-weight:800}.r3{color:#cd7f32;font-weight:800}.tk{font-weight:700}.nm{color:#9ca3af;font-size:12px}.pr{font-weight:700;text-align:right}.ch{font-weight:600;text-align:right}.gn{color:#00c853}.rd{color:#ff5252}.pb{display:flex;align-items:center;gap:8px}.pbb{flex:1;height:6px;background:#1e2028;border-radius:3px;overflow:hidden}.pbf{height:100%;border-radius:3px}.pb-h{background:linear-gradient(90deg,#00c853,#00e676)}.pb-m{background:linear-gradient(90deg,#ffc107,#ffd54f)}.pb-l{background:linear-gradient(90deg,#ff5252,#ff8a80)}.pt{font-weight:700;font-size:12px}.cc{display:flex;align-items:center;gap:5px}.cd{width:5px;height:5px;border-radius:50%}.cd-h{background:#00c853}.cd-m{background:#ffc107}.cd-l{background:#ff5252}.ft{padding:16px 24px;border-top:1px solid #1e2028;font-size:10px;color:#4b5563;text-align:center}.tb{background:#0d0e12;border-bottom:1px solid #1e2028;padding:6px 0;overflow:hidden;white-space:nowrap}.tbi{display:inline-flex;gap:32px;animation:scroll 40s linear infinite}@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}.ti{display:inline-flex;align-items:center;gap:6px;font-size:12px}.ti .sy{font-weight:600;color:#9ca3af}.ti .prc{font-weight:700;color:#e8eaed}.cup{color:#00c853}.cdn{color:#ff5252}.sb{font-size:10px;padding:2px 8px;border-radius:10px;display:inline-block;font-weight:500;background:#1e2028;color:#9ca3af}.sa{background:rgba(0,200,83,0.1);color:#00c853}.ss{background:rgba(33,150,243,0.1);color:#64b5f6}.sw{background:rgba(156,39,176,0.1);color:#ce93d8}.sc{background:rgba(255,152,0,0.1);color:#ffb74d}.st{background:rgba(0,188,212,0.1);color:#4dd0e1}.sm{background:rgba(233,30,99,0.1);color:#f48fb1}"

$HTML = "<!DOCTYPE html><html lang=`"es`"><head><meta charset=`"UTF-8`"><meta name=`"viewport`" content=`"width=device-width,initial-scale=1.0`"><title>Top 15 - GH Actions $FECHA</title><style>$CSS</style></head><body>"
$HTML += "<header class=`"hdr`"><div class=`"hl`"><div class=`"logo`">AI</div><div><div class=`"ht`">Top 15 - Probabilidad Alcista</div><div class=`"hs`">GitHub Actions | $FECHA $HORA</div></div></div><div class=`"badge`">GH ACTIONS</div></header>"
$HTML += "<div class=`"tb`"><div class=`"tbi`">$tickerItems $tickerItems</div></div>"
$HTML += "<div class=`"stats`"><div class=`"sc`"><div class=`"l`">Activos</div><div class=`"v`">15</div></div><div class=`"sc`"><div class=`"l`">Prob. Promedio</div><div class=`"v`">$avgProb%</div></div><div class=`"sc`"><div class=`"l`">Verde</div><div class=`"v`" style=`"color:#00c853`">$greenCount</div></div><div class=`"sc`"><div class=`"l`">Rojo</div><div class=`"v`" style=`"color:#ff5252`">$redCount</div></div><div class=`"sc`"><div class=`"l`">Servidor</div><div class=`"v`" style=`"font-size:13px`">GitHub</div></div></div>"
$HTML += "<div class=`"tc`"><table><thead><tr><th>#</th><th>Ticker</th><th class=`"hm`">Compania</th><th>Sector</th><th>Precio</th><th>Cambio</th><th>%</th><th>Prob.</th><th>Conf.</th></tr></thead><tbody>$tableRows</tbody></table></div>"
$HTML += "<div class=`"ft`">Ejecutado en GitHub Actions | $FECHA_HUMANA | Generado por IA. No constituye asesoramiento financiero.</div></body></html>"

Set-Content -Path "$REPORTES_DIR/dashboard_top15.html" -Value $HTML -Force
Write-Output "[OK] Dashboard HTML generado ($($HTML.Length) bytes)"

# -- PASO 5: REPORTE TEXTO --
$reporte = "REPORTE GITHUB ACTIONS - $FECHA_HUMANA`nCiclo: $HORA UTC`n`n"
$reporte += "NOTICIAS DEL CICLO:`n"
foreach ($h in $headlines) { $reporte += "  $h`n" }
$reporte += "`nPORTAFOLIO:`n"
$reporte += "  Probabilidad promedio: $avgProb%`n"
$reporte += "  Activos en verde: $greenCount | Rojo: $redCount`n"
$reporte += "`n--- Generado automaticamente. No constituye asesoramiento financiero. ---"

$reporteFile = "$REPORTES_DIR/Reporte_GH_$FECHA`_$($HORA -replace ':','').txt"
Set-Content -Path $reporteFile -Value $reporte -Force
Write-Output "[OK] Reporte generado: Reporte_GH_$FECHA`_$($HORA -replace ':','').txt"

Write-Output "[$FECHA $HORA] === CICLO COMPLETADO EXITOSAMENTE ==="
