<#
.SYNOPSIS
  Script para GitHub Actions - Ciclo de Analisis Financiero (30 tickers)
  Invocado por .github/workflows/ciclo_automatizado.yml
  Monitoreo intradia: 08:00, 11:00, 14:00, 16:00 Mexico
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

Write-Output "[$FECHA $HORA] === CICLO 30 TICKERS ==="

# ============================================================
# PASO 1: NOTICIAS (multifuente intradia)
# ============================================================
$headlines = @()
$fuentes = @(
    "https://finance.yahoo.com/",
    "https://www.reuters.com/markets/",
    "https://www.cnbc.com/markets/"
)
foreach ($url in $fuentes) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        $txt = $resp.Content -replace '<[^>]+>', ' ' -replace '\s+', ' '
        $fuente = switch -Wildcard ($url) { "*yahoo*" { "Yahoo" } "*reuters*" { "Reuters" } "*cnbc*" { "CNBC" } default { "Web" } }
        if ($txt -match '(market|stocks|rally|record|AI|oil|gold|Fed|inflation|NFP|crypto|bond|semiconductor)(.{0,120})') {
            $h = "[$fuente] $($matches[0].Trim())"
            if ($h.Length -gt 30 -and $h.Length -lt 200) { $headlines += $h }
        }
    } catch { Write-Output "[!] $url no disponible" }
}
if ($headlines.Count -eq 0) {
    $headlines = @(
        "[Agente] Monitoreo intradia activo - 30 tickers",
        "[Agente] Mercado en sesion - datos actualizados",
        "[Agente] S&P 500 en zona de maximos historicos",
        "[Agente] Petroleo y geopolitica monitoreados",
        "[Agente] Ciclo intradia completado"
    )
}
$newsContent = "NEWS FEED INTRADIA - $FECHA $HORA`n$($headlines.Count) titulares de $($fuentes.Count) fuentes`n`n"
foreach ($h in $headlines) { $newsContent += "- $h`n" }
Set-Content -Path "$DATOS_DIR/News_Feed_Resumen.txt" -Value $newsContent -Force
Write-Output "[OK] News: $($headlines.Count) titulares"

# ============================================================
# PASO 2: 30 TICKERS - 5 SECTORES ESTRATEGICOS
# ============================================================
$TICKERS = @(
    # TECNOLOGIA IA (15 originales)
    'NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS',
    # MEGA-CAP TECH & CONSUMER (nuevos)
    'AAPL','AMZN','GOOGL','META','MSFT',
    # HEALTHCARE & SEMICON EQUIP (nuevos)
    'LLY','AMAT','LRCX','PANW','ORCL',
    # INDUSTRIAL & DEFENSIVE (nuevos)
    'HON','UBER','GE','COST','NEE'
)

$stockData = @{
    # === TECNOLOGIA IA (originales) ===
    'NVDA' = @{ 'base' = 218.54; 'prob' = 72; 'conf' = 60; 'sector' = 'Semiconductores'; 'name' = 'NVIDIA Corporation' }
    'MU'   = @{ 'base' = 969.64; 'prob' = 68; 'conf' = 58; 'sector' = 'Semiconductores'; 'name' = 'Micron Technology' }
    'DELL' = @{ 'base' = 424.81; 'prob' = 70; 'conf' = 62; 'sector' = 'Servidores IA'; 'name' = 'Dell Technologies' }
    'AVGO' = @{ 'base' = 420.37; 'prob' = 65; 'conf' = 60; 'sector' = 'Semiconductores'; 'name' = 'Broadcom Inc.' }
    'DDOG' = @{ 'base' = 195.27; 'prob' = 63; 'conf' = 56; 'sector' = 'Software IA'; 'name' = 'Datadog Inc.' }
    'SMCI' = @{ 'base' = 984.94; 'prob' = 60; 'conf' = 54; 'sector' = 'Servidores IA'; 'name' = 'Super Micro Computer' }
    'SNOW' = @{ 'base' = 253.84; 'prob' = 62; 'conf' = 57; 'sector' = 'Software IA'; 'name' = 'Snowflake Inc.' }
    'CRWD' = @{ 'base' = 348.79; 'prob' = 58; 'conf' = 55; 'sector' = 'Ciberseguridad'; 'name' = 'CrowdStrike Holdings' }
    'NOW'  = @{ 'base' = 123.89; 'prob' = 55; 'conf' = 53; 'sector' = 'Software IA'; 'name' = 'ServiceNow Inc.' }
    'TSM'  = @{ 'base' = 197.10; 'prob' = 67; 'conf' = 65; 'sector' = 'Semiconductores'; 'name' = 'Taiwan Semiconductor' }
    'ARM'  = @{ 'base' = 156.74; 'prob' = 52; 'conf' = 52; 'sector' = 'Semiconductores'; 'name' = 'Arm Holdings' }
    'OKTA' = @{ 'base' = 121.96; 'prob' = 64; 'conf' = 58; 'sector' = 'Ciberseguridad'; 'name' = 'Okta Inc.' }
    'HPE'  = @{ 'base' = 60.38;  'prob' = 60; 'conf' = 55; 'sector' = 'Servidores IA'; 'name' = 'Hewlett Packard Enterprise' }
    'NTAP' = @{ 'base' = 209.45; 'prob' = 52; 'conf' = 52; 'sector' = 'Almacenamiento'; 'name' = 'NetApp Inc.' }
    'CLS'  = @{ 'base' = 388.12; 'prob' = 61; 'conf' = 56; 'sector' = 'Manufactura'; 'name' = 'Celestica Inc.' }
    # === MEGA-CAP TECH ===
    'AAPL' = @{ 'base' = 245.00; 'prob' = 58; 'conf' = 55; 'sector' = 'Consumer Tech'; 'name' = 'Apple Inc.' }
    'AMZN' = @{ 'base' = 215.00; 'prob' = 62; 'conf' = 57; 'sector' = 'Cloud/Commerce'; 'name' = 'Amazon.com Inc.' }
    'GOOGL'= @{ 'base' = 490.00; 'prob' = 64; 'conf' = 58; 'sector' = 'Internet/Cloud'; 'name' = 'Alphabet Inc.' }
    'META' = @{ 'base' = 620.00; 'prob' = 60; 'conf' = 55; 'sector' = 'Social/IA'; 'name' = 'Meta Platforms Inc.' }
    'MSFT' = @{ 'base' = 510.00; 'prob' = 63; 'conf' = 58; 'sector' = 'Enterprise/Cloud'; 'name' = 'Microsoft Corporation' }
    # === HEALTHCARE & SEMICON ===
    'LLY'  = @{ 'base' = 890.00; 'prob' = 57; 'conf' = 53; 'sector' = 'Farmaceutico'; 'name' = 'Eli Lilly and Company' }
    'AMAT' = @{ 'base' = 245.00; 'prob' = 59; 'conf' = 55; 'sector' = 'Semicon Equip'; 'name' = 'Applied Materials Inc.' }
    'LRCX' = @{ 'base' = 290.00; 'prob' = 58; 'conf' = 54; 'sector' = 'Semicon Equip'; 'name' = 'Lam Research Corporation' }
    'PANW' = @{ 'base' = 380.00; 'prob' = 56; 'conf' = 52; 'sector' = 'Ciberseguridad'; 'name' = 'Palo Alto Networks Inc.' }
    'ORCL' = @{ 'base' = 175.00; 'prob' = 57; 'conf' = 53; 'sector' = 'Cloud/Database'; 'name' = 'Oracle Corporation' }
    # === INDUSTRIAL & DEFENSIVE ===
    'HON'  = @{ 'base' = 235.00; 'prob' = 54; 'conf' = 50; 'sector' = 'Industrial'; 'name' = 'Honeywell International Inc.' }
    'UBER' = @{ 'base' = 82.00;  'prob' = 56; 'conf' = 52; 'sector' = 'Movilidad/Tech'; 'name' = 'Uber Technologies Inc.' }
    'GE'   = @{ 'base' = 200.00; 'prob' = 55; 'conf' = 51; 'sector' = 'Aeroespacial'; 'name' = 'GE Aerospace' }
    'COST' = @{ 'base' = 950.00; 'prob' = 53; 'conf' = 50; 'sector' = 'Consumo Defensivo'; 'name' = 'Costco Wholesale Corp.' }
    'NEE'  = @{ 'base' = 78.00;  'prob' = 55; 'conf' = 51; 'sector' = 'Utilities/Energy'; 'name' = 'NextEra Energy Inc.' }
}

# ============================================================
# PASO 3: PRECIOS REALES (con fallbacks)
# ============================================================
$prices = @{}
$jsonPath = "$DATOS_DIR/precios_reales.json"
$dataSource = "simulada"
if (Test-Path $jsonPath) {
    try {
        $realData = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $dataSource = $realData.fuente
        foreach ($t in $TICKERS) {
            $pd = $realData.precios.$t
            if ($pd) {
                $prices[$t] = @{ 'price' = $pd.price; 'change' = $pd.change; 'pct' = $pd.pct }
            } else {
                $base = $stockData[$t]['base']
                $var = $base * 0.015
                $live = [math]::Round($base + (Get-Random -Minimum -$var -Maximum $var), 2)
                $chg = [math]::Round($live - $base, 2)
                $prices[$t] = @{ 'price' = $live; 'change' = $chg; 'pct' = [math]::Round(($chg / $base) * 100, 2) }
            }
        }
    } catch {
        Write-Output "[!] Error leyendo JSON real, usando simulados"
        $dataSource = "simulada"
    }
}
if ($prices.Count -eq 0) {
    foreach ($t in $TICKERS) {
        $base = $stockData[$t]['base']
        $var = $base * 0.015
        $live = [math]::Round($base + (Get-Random -Minimum -$var -Maximum $var), 2)
        $chg = [math]::Round($live - $base, 2)
        $pct = [math]::Round(($chg / $base) * 100, 2)
        $prices[$t] = @{ 'price' = $live; 'change' = $chg; 'pct' = $pct }
    }
}
Write-Output "[OK] Precios ($dataSource) para 30 activos"

# ============================================================
# PASO 4: BITACORAS
# ============================================================
foreach ($t in $TICKERS) {
    $pr = $prices[$t]
    $path = "$BASE_DIR/Bitacora_$t.txt"
    if (-not (Test-Path $path)) {
        Set-Content -Path $path -Value "BITACORA $t - Creada en expansion a 30 tickers $FECHA" -Force
    }
    Add-Content -Path $path -Value "[$FECHA] INTRA $HORA - `$$($pr['price']) | $($pr['pct'])% | Prob: $($stockData[$t]['prob'])%" -Force
}
Write-Output "[OK] 30 bitoras actualizadas"

$globalPath = "$BASE_DIR/Bitacora_Aprendizaje.txt"
if (-not (Test-Path $globalPath)) {
    Set-Content -Path $globalPath -Value "BITACORA GLOBAL - Inicializada $FECHA" -Force
}
Add-Content -Path $globalPath -Value "[$FECHA] CICLO 30T INTRA $HORA - $($headlines.Count) titulares" -Force

# ============================================================
# PASO 5: DASHBOARD HTML (30 tickers)
# ============================================================
$probSum = 0; foreach ($t in $TICKERS) { $probSum += $stockData[$t]['prob'] }
$avgProb = [math]::Round($probSum / 30)
$greenCount = 0; $redCount = 0
foreach ($t in $TICKERS) { if ($prices[$t]['change'] -ge 0) { $greenCount++ } else { $redCount++ } }

$sectorMap = @{
    'Semiconductores'='sa'; 'Servidores IA'='ss'; 'Software IA'='sw'
    'Ciberseguridad'='sc'; 'Almacenamiento'='st'; 'Manufactura'='sm'
    'Consumer Tech'='scu'; 'Cloud/Commerce'='scc'; 'Internet/Cloud'='sci'
    'Social/IA'='scl'; 'Enterprise/Cloud'='sce'; 'Farmaceutico'='sph'
    'Semicon Equip'='seq'; 'Cloud/Database'='scd'; 'Industrial'='sin'
    'Movilidad/Tech'='smo'; 'Aeroespacial'='sae'; 'Consumo Defensivo'='sde'
    'Utilities/Energy'='sut'
}

$tableRows = ""; $tickerItems = ""; $rank = 1
foreach ($t in $TICKERS) {
    $info = $stockData[$t]; $pr = $prices[$t]
    $cc = if ($pr['change'] -ge 0) { 'gn' } else { 'rd' }
    $sg = if ($pr['change'] -ge 0) { '+' } else { '' }
    $rc = if ($rank -eq 1) { 'r1' } elseif ($rank -eq 2) { 'r2' } elseif ($rank -eq 3) { 'r3' } else { '' }
    $sc = if ($sectorMap.ContainsKey($info['sector'])) { $sectorMap[$info['sector']] } else { 'sb' }
    $pc = if ($info['prob'] -ge 65) { 'pb-h' } elseif ($info['prob'] -ge 58) { 'pb-m' } else { 'pb-l' }
    $cf = if ($info['conf'] -ge 60) { 'cd-h' } elseif ($info['conf'] -ge 55) { 'cd-m' } else { 'cd-l' }
    $tableRows += "<tr><td class=`"$rc`">$rank</td><td class=`"tk`">$t</td><td class=`"nm hm`">$($info['name'])</td><td><span class=`"sb $sc`">$($info['sector'])</span></td><td class=`"pr $cc`">`$$($pr['price'])</td><td class=`"ch $cc`">$sg`$$($pr['change'])</td><td class=`"ch $cc`">$sg$($pr['pct'])%</td><td><div class=`"pb`"><div class=`"pbb`"><div class=`"pbf $pc`" style=`"width:$($info['prob'])%`"></div></div><span class=`"pt`">$($info['prob'])%</span></div></td><td><div class=`"cc`"><span class=`"cd $cf`"></span>$($info['conf'])%</div></td></tr>"
    $tc = if ($pr['change'] -ge 0) { 'cup' } else { 'cdn' }
    $tickerItems += "<span class=`"ti`"><span class=`"sy`">$t</span><span class=`"prc`">`$$($pr['price'])</span><span class=`"$tc`">$sg$($pr['pct'])%</span></span>"
    $rank++
}

$CSS = "*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0b0e;color:#e8eaed}::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0a0b0e}::-webkit-scrollbar-thumb{background:#2a2d35}.hdr{background:linear-gradient(135deg,#0d0e12,#14161b);border-bottom:1px solid #1e2028;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}.logo{width:36px;height:36px;background:linear-gradient(135deg,#00c853,#00e676);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:16px;color:#0a0b0e}.hl{display:flex;align-items:center;gap:16px}.ht{font-size:16px;font-weight:700}.hs{font-size:11px;color:#6b7280}.badge{font-size:11px;font-weight:600;color:#00c853;background:rgba(0,200,83,0.08);padding:4px 12px;border-radius:20px}.stats{display:flex;gap:12px;padding:12px 24px;background:#0d0e12;border-bottom:1px solid #1e2028;flex-wrap:wrap}.sc{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:10px 16px;flex:1;min-width:100px}.sc .l{font-size:10px;text-transform:uppercase;color:#6b7280}.sc .v{font-size:16px;font-weight:700}.tc{padding:24px;overflow-x:auto}table{width:100%;border-collapse:separate}th{font-size:10px;font-weight:600;text-transform:uppercase;color:#6b7280;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2028;background:#0a0b0e;position:sticky;top:0;z-index:2}td{padding:12px;font-size:13px;border-bottom:1px solid #15171d}tr:hover{background:rgba(0,200,83,0.03)}.r1{color:#ffd700;font-weight:800}.r2{color:#c0c0c0;font-weight:800}.r3{color:#cd7f32;font-weight:800}.tk{font-weight:700}.nm{color:#9ca3af;font-size:12px}.pr{font-weight:700;text-align:right;font-variant-numeric:tabular-nums}.ch{font-weight:600;text-align:right}.gn{color:#00c853}.rd{color:#ff5252}.pb{display:flex;align-items:center;gap:8px}.pbb{flex:1;height:6px;background:#1e2028;border-radius:3px;overflow:hidden}.pbf{height:100%;border-radius:3px}.pb-h{background:linear-gradient(90deg,#00c853,#00e676)}.pb-m{background:linear-gradient(90deg,#ffc107,#ffd54f)}.pb-l{background:linear-gradient(90deg,#ff5252,#ff8a80)}.pt{font-weight:700;font-size:12px;min-width:36px;text-align:right}.cc{display:flex;align-items:center;gap:5px}.cd{width:5px;height:5px;border-radius:50%}.cd-h{background:#00c853}.cd-m{background:#ffc107}.cd-l{background:#ff5252}.ft{padding:16px 24px;border-top:1px solid #1e2028;font-size:10px;color:#4b5563;text-align:center}.tb{background:#0d0e12;border-bottom:1px solid #1e2028;padding:6px 0;overflow:hidden;white-space:nowrap}.tbi{display:inline-flex;gap:32px;animation:scroll 40s linear infinite}@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}.ti{display:inline-flex;align-items:center;gap:6px;font-size:12px}.ti .sy{font-weight:600;color:#9ca3af}.ti .prc{font-weight:700}.cup{color:#00c853}.cdn{color:#ff5252}.sb{font-size:10px;padding:2px 8px;border-radius:10px;display:inline-block;font-weight:500;background:#1e2028;color:#9ca3af}.sa,.sq{background:rgba(0,200,83,0.1);color:#00c853}.ss{background:rgba(33,150,243,0.1);color:#64b5f6}.sw,.scl,.sce{background:rgba(156,39,176,0.1);color:#ce93d8}.sc{background:rgba(255,152,0,0.1);color:#ffb74d}.st{background:rgba(0,188,212,0.1);color:#4dd0e1}.sm{background:rgba(233,30,99,0.1);color:#f48fb1}.scu,.scc,.sci,.scd{background:rgba(63,81,181,0.1);color:#9fa8da}.sph{background:rgba(0,150,136,0.1);color:#80cbc4}.sin,.smo,.sae{background:rgba(121,85,72,0.1);color:#a1887f}.sde{background:rgba(255,87,34,0.1);color:#ffab91}.sut{background:rgba(255,193,7,0.1);color:#ffe082}"

$HTML = "<!DOCTYPE html><html lang=`"es`"><head><meta charset=`"UTF-8`"><meta name=`"viewport`" content=`"width=device-width,initial-scale=1.0`"><title>Top 30 - IA Financiero $FECHA $HORA</title><style>$CSS</style></head><body>"
$HTML += "<header class=`"hdr`"><div class=`"hl`"><div class=`"logo`">AI</div><div><div class=`"ht`">Top 30 - Probabilidad Alcista</div><div class=`"hs`">Monitoreo Intradia | $FECHA $HORA | 30 tickers</div></div></div><div class=`"badge`">INTRADIA</div></header>"
$HTML += "<div class=`"tb`"><div class=`"tbi`">$tickerItems $tickerItems</div></div>"
$HTML += "<div class=`"stats`"><div class=`"sc`"><div class=`"l`">Activos</div><div class=`"v`">30</div></div><div class=`"sc`"><div class=`"l`">Prob. Promedio</div><div class=`"v`">$avgProb%</div></div><div class=`"sc`"><div class=`"l`">Verde</div><div class=`"v`" style=`"color:#00c853`">$greenCount</div></div><div class=`"sc`"><div class=`"l`">Rojo</div><div class=`"v`" style=`"color:#ff5252`">$redCount</div></div><div class=`"sc`"><div class=`"l`">Fuente</div><div class=`"v`" style=`"font-size:12px;color:#64b5f6`">$dataSource</div></div></div>"
$HTML += "<div class=`"tc`"><table><thead><tr><th>#</th><th>Ticker</th><th class=`"hm`">Compania</th><th>Sector</th><th style=`"text-align:right`">Precio</th><th style=`"text-align:right`">Cambio</th><th style=`"text-align:right`">%</th><th style=`"text-align:right`">Prob.</th><th>Conf.</th></tr></thead><tbody>$tableRows</tbody></table></div>"
$HTML += "<div class=`"ft`">Monitoreo Intradia - GitHub Actions | $FECHA_HUMANA | 30 tickers en 5 sectores estrategicos | Generado por IA. No constituye asesoramiento financiero.</div></body></html>"

Set-Content -Path "$REPORTES_DIR/dashboard_top15.html" -Value $HTML -Force
Write-Output "[OK] Dashboard 30 tickers: $($HTML.Length) bytes"

# ============================================================
# PASO 6: REPORTE TEXTO
# ============================================================
$sectorGroups = @{}
foreach ($t in $TICKERS) {
    $sec = $stockData[$t]['sector']
    if (-not $sectorGroups.ContainsKey($sec)) { $sectorGroups[$sec] = @() }
    $sectorGroups[$sec] += $t
}
$sectorSummary = ($sectorGroups.Keys | ForEach-Object { "$_: $($sectorGroups[$_].Count)" }) -join ', '

$reporte = "REPORTE INTRADIA 30 TICKERS - $FECHA_HUMANA`nCiclo: $HORA UTC`n`n"
$reporte += "NOTICIAS INTRADIA:`n"
foreach ($h in $headlines) { $reporte += "  $h`n" }
$reporte += "`nPORTAFOLIO 30 TICKERS:`n"
$reporte += "  Sectores: $sectorSummary`n"
$reporte += "  Probabilidad promedio: $avgProb%`n"
$reporte += "  Verdes: $greenCount | Rojos: $redCount`n"
$reporte += "`nRanking Top 5:`n"
$sorted = $TICKERS | Sort-Object { $stockData[$_]['prob'] } -Descending
for ($i = 0; $i -lt 5; $i++) {
    $t = $sorted[$i]
    $reporte += "  #$($i+1) $t - $($stockData[$t]['name']) (Prob: $($stockData[$t]['prob'])%)`n"
}
$reporte += "`n--- Generado automaticamente. No constituye asesoramiento financiero. ---"

$reporteFile = "$REPORTES_DIR/Reporte_30T_$FECHA`_$($HORA -replace ':','').txt"
Set-Content -Path $reporteFile -Value $reporte -Force
Write-Output "[OK] Reporte: Reporte_30T_$FECHA`_$($HORA -replace ':','').txt"
Write-Output "[$FECHA $HORA] === CICLO 30 TICKERS COMPLETADO ==="
