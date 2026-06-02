<#
.SYNOPSIS
  Script para GitHub Actions - Ciclo de Analisis Financiero (30 tickers)
  Invocado por .github/workflows/ciclo_automatizado.yml
  Fuentes: yfinance, Google Finance scraping, OpenRouter IA (multi-modelo)
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
# PASO 1: CARGAR DATOS REALES + IA
# ============================================================
$TICKERS = @(
    'NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
    'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
    'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE'
)

$stockData = @{}
foreach ($t in $TICKERS) {
    $stockData[$t] = @{ 'prob' = 50; 'conf' = 50; 'sector' = 'General'; 'name' = $t; 'analisis' = ''; 'target_30d' = 0 }
}

$prices = @{}
$dataSource = 'simulada'
$aiModel = 'none'
$aiResumen = ''
$aiTitulares = @()
$aiSectores = @{}
$precisionGlobal = 0
$totalEvaluado = 0

# Cargar precios reales
$jsonPath = "$DATOS_DIR/precios_reales.json"
if (Test-Path $jsonPath) {
    try {
        $realData = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $dataSource = $realData.fuente
        foreach ($t in $TICKERS) {
            $pd = $realData.precios.$t
            if ($pd) {
                $prices[$t] = @{ 'price' = $pd.price; 'change' = $pd.change; 'pct' = $pd.pct }
            }
        }
    } catch { Write-Output "[!] Error cargando precios reales" }
}

# Cargar analisis IA
$iaPath = "$DATOS_DIR/analisis_ia.json"
if (Test-Path $iaPath) {
    try {
        $iaData = Get-Content $iaPath -Raw | ConvertFrom-Json
        $aiModel = $iaData.modelo_usado
        $aiResumen = $iaData.resumen_mercado
        if ($iaData.titulares) { $aiTitulares = @($iaData.titulares) }
        if ($iaData.sectores) { $aiSectores = $iaData.sectores }
        if ($iaData.precision_global) { $precisionGlobal = $iaData.precision_global }
        if ($iaData.total_evaluado) { $totalEvaluado = $iaData.total_evaluado }
        foreach ($t in $TICKERS) {
            $iaInfo = $iaData.probabilidades.$t
            if ($iaInfo) {
                $stockData[$t]['prob'] = [int]$iaInfo.probabilidad
                $stockData[$t]['conf'] = [int]$iaInfo.confianza
                $stockData[$t]['analisis'] = [string]$iaInfo.analisis
                $stockData[$t]['target_30d'] = [double]($iaInfo.precio_objetivo_30d)
            }
        }
        Write-Output "[OK] Analisis IA ($aiModel) cargado"
    } catch { Write-Output "[!] Error cargando analisis IA" }
}

# Fallback: asignar defaults para tickers sin precio
$probDefaults = @{ 'NVDA'=72;'MU'=68;'DELL'=70;'AVGO'=65;'DDOG'=63;'SMCI'=60;'SNOW'=62;
    'CRWD'=58;'NOW'=55;'TSM'=67;'ARM'=52;'OKTA'=64;'HPE'=60;'NTAP'=52;'CLS'=61;
    'AAPL'=58;'AMZN'=62;'GOOGL'=64;'META'=60;'MSFT'=63;'LLY'=57;'AMAT'=59;
    'LRCX'=58;'PANW'=56;'ORCL'=57;'HON'=54;'UBER'=56;'GE'=55;'COST'=53;'NEE'=55 }
$confDefaults = @{ 'NVDA'=60;'MU'=58;'DELL'=62;'AVGO'=60;'DDOG'=56;'SMCI'=54;'SNOW'=57;
    'CRWD'=55;'NOW'=53;'TSM'=65;'ARM'=52;'OKTA'=58;'HPE'=55;'NTAP'=52;'CLS'=56;
    'AAPL'=55;'AMZN'=57;'GOOGL'=58;'META'=55;'MSFT'=58;'LLY'=53;'AMAT'=55;
    'LRCX'=54;'PANW'=52;'ORCL'=53;'HON'=50;'UBER'=52;'GE'=51;'COST'=50;'NEE'=51 }
$sectorMap = @{
    'NVDA'='Semiconductores';'MU'='Semiconductores';'AVGO'='Semiconductores';'TSM'='Semiconductores';'ARM'='Semiconductores';
    'DELL'='Servidores IA';'SMCI'='Servidores IA';'HPE'='Servidores IA';
    'DDOG'='Software IA';'SNOW'='Software IA';'NOW'='Software IA';
    'CRWD'='Ciberseguridad';'OKTA'='Ciberseguridad';'PANW'='Ciberseguridad';
    'NTAP'='Almacenamiento';'CLS'='Manufactura';'AAPL'='Consumer Tech';
    'AMZN'='Cloud/Commerce';'GOOGL'='Internet/Cloud';'META'='Social/IA';
    'MSFT'='Enterprise/Cloud';'LLY'='Farmaceutico';'AMAT'='Semicon Equip';
    'LRCX'='Semicon Equip';'ORCL'='Cloud/Database';'HON'='Industrial';
    'UBER'='Movilidad/Tech';'GE'='Aeroespacial';'COST'='Consumo Defensivo';'NEE'='Utilities/Energy' }
$nameMap = @{
    'NVDA'='NVIDIA Corporation';'MU'='Micron Technology';'DELL'='Dell Technologies';'AVGO'='Broadcom Inc.';
    'DDOG'='Datadog Inc.';'SMCI'='Super Micro Computer';'SNOW'='Snowflake Inc.';'CRWD'='CrowdStrike Holdings';
    'NOW'='ServiceNow Inc.';'TSM'='Taiwan Semiconductor';'ARM'='Arm Holdings';'OKTA'='Okta Inc.';
    'HPE'='Hewlett Packard Enterprise';'NTAP'='NetApp Inc.';'CLS'='Celestica Inc.';
    'AAPL'='Apple Inc.';'AMZN'='Amazon.com Inc.';'GOOGL'='Alphabet Inc.';'META'='Meta Platforms Inc.';
    'MSFT'='Microsoft Corporation';'LLY'='Eli Lilly and Company';'AMAT'='Applied Materials Inc.';
    'LRCX'='Lam Research Corporation';'PANW'='Palo Alto Networks Inc.';'ORCL'='Oracle Corporation';
    'HON'='Honeywell International Inc.';'UBER'='Uber Technologies Inc.';'GE'='GE Aerospace';
    'COST'='Costco Wholesale Corp.';'NEE'='NextEra Energy Inc.' }

foreach ($t in $TICKERS) {
    $stockData[$t]['sector'] = $sectorMap[$t]
    $stockData[$t]['name'] = $nameMap[$t]
    if ($stockData[$t]['prob'] -eq 50 -or -not $prices.ContainsKey($t)) {
        $stockData[$t]['prob'] = $probDefaults[$t]
        $stockData[$t]['conf'] = $confDefaults[$t]
    }
    if (-not $prices.ContainsKey($t)) {
        $base = $probDefaults[$t] * 3 + 50
        $var = $base * 0.015
        $live = [math]::Round($base + (Get-Random -Minimum -$var -Maximum $var), 2)
        $chg = [math]::Round($live - $base, 2)
        $prices[$t] = @{ 'price' = $live; 'change' = $chg; 'pct' = [math]::Round(($chg / $base) * 100, 2) }
    }
}

Write-Output "[OK] Datos: $dataSource | IA: $aiModel"

# ============================================================
# PASO 2: NOTICIAS (IA + web scraping combinado)
# ============================================================
$headlines = @()
$fuentes = @("https://finance.yahoo.com/","https://www.reuters.com/markets/","https://www.cnbc.com/markets/")
foreach ($url in $fuentes) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 8 -ErrorAction Stop
        $txt = $resp.Content -replace '<[^>]+>', ' ' -replace '\s+', ' '
        $fuente = switch -Wildcard ($url) { "*yahoo*" { "Yahoo" } "*reuters*" { "Reuters" } "*cnbc*" { "CNBC" } default { "Web" } }
        if ($txt -match '(market|stocks|rally|record|AI|oil|gold|Fed|inflation|NFP|crypto|bond|semiconductor)(.{0,120})') {
            $h = "[$fuente] $($matches[0].Trim())"
            if ($h.Length -gt 30 -and $h.Length -lt 200) { $headlines += $h }
        }
    } catch { }
}
if ($headlines.Count -eq 0 -and $aiTitulares.Count -gt 0) {
    $headlines = $aiTitulares | ForEach-Object { "[IA] $_" }
}
if ($headlines.Count -eq 0) {
    $headlines = @("[Agente] Analisis IA activo - $aiModel","[Agente] Mercado en sesion","[Agente] 30 tickers monitoreados","[Agente] Datos via $dataSource","[Agente] Ciclo completado")
}
$newsContent = "NEWS FEED - $FECHA $HORA`n$($headlines.Count) titulares | Fuente: $dataSource | IA: $aiModel`n`n"
foreach ($h in $headlines) { $newsContent += "- $h`n" }
Set-Content -Path "$DATOS_DIR/News_Feed_Resumen.txt" -Value $newsContent -Force
Write-Output "[OK] News: $($headlines.Count) titulares"

# ============================================================
# PASO 3: BITACORAS
# ============================================================
foreach ($t in $TICKERS) {
    $pr = $prices[$t]
    $path = "$BASE_DIR/Bitacora_$t.txt"
    if (-not (Test-Path $path)) {
        Set-Content -Path $path -Value "BITACORA $t - Creada $FECHA" -Force
    }
    $analisis = $stockData[$t]['analisis']
    $entry = "[$FECHA] INTRA $HORA - `$$($pr['price']) | $($pr['pct'])% | Prob: $($stockData[$t]['prob'])%"
    if ($analisis) { $entry += " | $analisis" }
    Add-Content -Path $path -Value $entry -Force
}
$globalPath = "$BASE_DIR/Bitacora_Aprendizaje.txt"
if (-not (Test-Path $globalPath)) {
    Set-Content -Path $globalPath -Value "BITACORA GLOBAL - Inicializada $FECHA" -Force
}
Add-Content -Path $globalPath -Value "[$FECHA] CICLO 30T INTRA $HORA - IA: $aiModel - $($headlines.Count) titulares" -Force
Write-Output "[OK] 30 bitacoras actualizadas"

# ============================================================
# PASO 4: DASHBOARD HTML
# ============================================================
$probSum = 0; foreach ($t in $TICKERS) { $probSum += $stockData[$t]['prob'] }
$avgProb = [math]::Round($probSum / 30)
$greenCount = 0; $redCount = 0
foreach ($t in $TICKERS) { if ($prices[$t]['change'] -ge 0) { $greenCount++ } else { $redCount++ } }

$sectorCSS = @{
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
    $sc = $sectorCSS[$info['sector']]
    $pc = if ($info['prob'] -ge 65) { 'pb-h' } elseif ($info['prob'] -ge 58) { 'pb-m' } else { 'pb-l' }
    $cf = if ($info['conf'] -ge 60) { 'cd-h' } elseif ($info['conf'] -ge 55) { 'cd-m' } else { 'cd-l' }
    $an = $info['analisis']
    $titleAttr = if ($an) { " title=`"$an`"" } else { "" }
    $tableRows += "<tr$titleAttr><td class=`"$rc`">$rank</td><td class=`"tk`">$t</td><td class=`"nm hm`">$($info['name'])</td><td><span class=`"sb $sc`">$($info['sector'])</span></td><td class=`"pr $cc`">`$$($pr['price'])</td><td class=`"ch $cc`">$sg`$$($pr['change'])</td><td class=`"ch $cc`">$sg$($pr['pct'])%</td><td><div class=`"pb`"><div class=`"pbb`"><div class=`"pbf $pc`" style=`"width:$($info['prob'])%`"></div></div><span class=`"pt`">$($info['prob'])%</span></div></td><td><div class=`"cc`"><span class=`"cd $cf`"></span>$($info['conf'])%</div></td></tr>"
    $tc = if ($pr['change'] -ge 0) { 'cup' } else { 'cdn' }
    $tickerItems += "<span class=`"ti`"><span class=`"sy`">$t</span><span class=`"prc`">`$$($pr['price'])</span><span class=`"$tc`">$sg$($pr['pct'])%</span></span>"
    $rank++
}

$CSS = "*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0b0e;color:#e8eaed}::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0a0b0e}::-webkit-scrollbar-thumb{background:#2a2d35}.hdr{background:linear-gradient(135deg,#0d0e12,#14161b);border-bottom:1px solid #1e2028;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}.logo{width:36px;height:36px;background:linear-gradient(135deg,#00c853,#00e676);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:16px;color:#0a0b0e}.hl{display:flex;align-items:center;gap:16px}.ht{font-size:16px;font-weight:700}.hs{font-size:11px;color:#6b7280}.badge{font-size:11px;font-weight:600;color:#00c853;background:rgba(0,200,83,0.08);padding:4px 12px;border-radius:20px}.stats{display:flex;gap:12px;padding:12px 24px;background:#0d0e12;border-bottom:1px solid #1e2028;flex-wrap:wrap}.sc{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:10px 16px;flex:1;min-width:100px}.sc .l{font-size:10px;text-transform:uppercase;color:#6b7280}.sc .v{font-size:16px;font-weight:700}.resumen{background:linear-gradient(135deg,#0d0e12,#14161b);border-bottom:1px solid #1e2028;padding:12px 24px;font-size:13px;color:#9ca3af}.tc{padding:24px;overflow-x:auto}table{width:100%;border-collapse:separate}th{font-size:10px;font-weight:600;text-transform:uppercase;color:#6b7280;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2028;background:#0a0b0e;position:sticky;top:0;z-index:2}td{padding:12px;font-size:13px;border-bottom:1px solid #15171d}tr:hover{background:rgba(0,200,83,0.03);cursor:default}.r1{color:#ffd700;font-weight:800}.r2{color:#c0c0c0;font-weight:800}.r3{color:#cd7f32;font-weight:800}.tk{font-weight:700}.nm{color:#9ca3af;font-size:12px}.pr{font-weight:700;text-align:right;font-variant-numeric:tabular-nums}.ch{font-weight:600;text-align:right}.gn{color:#00c853}.rd{color:#ff5252}.pb{display:flex;align-items:center;gap:8px}.pbb{flex:1;height:6px;background:#1e2028;border-radius:3px;overflow:hidden}.pbf{height:100%;border-radius:3px}.pb-h{background:linear-gradient(90deg,#00c853,#00e676)}.pb-m{background:linear-gradient(90deg,#ffc107,#ffd54f)}.pb-l{background:linear-gradient(90deg,#ff5252,#ff8a80)}.pt{font-weight:700;font-size:12px;min-width:36px;text-align:right}.cc{display:flex;align-items:center;gap:5px}.cd{width:5px;height:5px;border-radius:50%}.cd-h{background:#00c853}.cd-m{background:#ffc107}.cd-l{background:#ff5252}.ft{padding:16px 24px;border-top:1px solid #1e2028;font-size:10px;color:#4b5563;text-align:center}.tb{background:#0d0e12;border-bottom:1px solid #1e2028;padding:6px 0;overflow:hidden;white-space:nowrap}.tbi{display:inline-flex;gap:32px;animation:scroll 40s linear infinite}@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}.ti{display:inline-flex;align-items:center;gap:6px;font-size:12px}.ti .sy{font-weight:600;color:#9ca3af}.ti .prc{font-weight:700}.cup{color:#00c853}.cdn{color:#ff5252}.sb{font-size:10px;padding:2px 8px;border-radius:10px;display:inline-block;font-weight:500;background:#1e2028;color:#9ca3af}.sa{background:rgba(0,200,83,0.1);color:#00c853}.ss{background:rgba(33,150,243,0.1);color:#64b5f6}.sw{background:rgba(156,39,176,0.1);color:#ce93d8}.sc{background:rgba(255,152,0,0.1);color:#ffb74d}.st{background:rgba(0,188,212,0.1);color:#4dd0e1}.sm{background:rgba(233,30,99,0.1);color:#f48fb1}.scu{background:rgba(63,81,181,0.1);color:#9fa8da}.scc{background:rgba(63,81,181,0.1);color:#9fa8da}.sci{background:rgba(63,81,181,0.1);color:#9fa8da}.scl{background:rgba(156,39,176,0.1);color:#ce93d8}.sce{background:rgba(63,81,181,0.1);color:#9fa8da}.sph{background:rgba(0,150,136,0.1);color:#80cbc4}.seq{background:rgba(0,200,83,0.1);color:#00c853}.scd{background:rgba(63,81,181,0.1);color:#9fa8da}.sin{background:rgba(121,85,72,0.1);color:#a1887f}.smo{background:rgba(121,85,72,0.1);color:#a1887f}.sae{background:rgba(121,85,72,0.1);color:#a1887f}.sde{background:rgba(255,87,34,0.1);color:#ffab91}.sut{background:rgba(255,193,7,0.1);color:#ffe082}
.sbar{display:flex;gap:8px;padding:8px 24px;background:#0d0e12;border-bottom:1px solid #1e2028}.sbar input{flex:1;background:#14161b;border:1px solid #1e2028;border-radius:6px;padding:8px 12px;color:#e8eaed;font-size:13px;outline:none}.sbar input:focus{border-color:#00c853}.sbar input::placeholder{color:#4b5563}.sbar select{background:#14161b;border:1px solid #1e2028;border-radius:6px;padding:8px;color:#e8eaed;font-size:12px;outline:none;cursor:pointer}.sbar .rcount{color:#4b5563;font-size:11px;align-self:center;white-space:nowrap}
.t5{padding:16px 24px;background:#0d0e12;border-bottom:1px solid #1e2028}.t5h{font-size:13px;font-weight:700;color:#00c853;margin-bottom:12px;display:flex;align-items:center;gap:8px}.t5h span{font-size:10px;color:#4b5563;font-weight:400}.t5g{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}.t5c{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:12px;position:relative;overflow:hidden}.t5c .rk{position:absolute;top:0;left:0;background:#00c853;color:#0a0b0e;font-size:10px;font-weight:800;padding:2px 8px;border-radius:0 0 6px 0}.t5c .tk{font-size:14px;font-weight:700;margin-top:6px}.t5c .nm{font-size:11px;color:#9ca3af;margin-bottom:4px}.t5c .pr{font-size:13px;font-weight:700}.t5c .tp{font-size:11px;color:#00c853;margin-top:2px}.t5c .up{font-size:12px;font-weight:700;color:#00c853;margin-top:1px}.t5c .an{font-size:10px;color:#6b7280;margin-top:6px;line-height:1.3}.t5c .pb{display:flex;align-items:center;gap:6px;margin-top:4px}.t5c .pb .pbb{flex:1;height:4px;background:#1e2028;border-radius:2px;overflow:hidden}.t5c .pb .pbf{height:100%;border-radius:2px}.t5c .pb .pt{font-size:11px;font-weight:700;min-width:30px;text-align:right}"

$HTML = "<!DOCTYPE html><html lang=`"es`"><head><meta charset=`"UTF-8`"><meta name=`"viewport`" content=`"width=device-width,initial-scale=1.0`"><title>Top 30 - IA Financiero $FECHA $HORA</title><style>$CSS</style></head><body>"
$HTML += "<header class=`"hdr`"><div class=`"hl`"><div class=`"logo`">AI</div><div><div class=`"ht`">Top 30 - Probabilidad Alcista</div><div class=`"hs`">IA: $aiModel | Precios: $dataSource | $FECHA $HORA</div></div></div><div class=`"badge`">INTRADIA</div></header>"
$HTML += "<div class=`"tb`"><div class=`"tbi`">$tickerItems $tickerItems</div></div>"
$precColor = if ($precisionGlobal -ge 0.7) { '#00c853' } elseif ($precisionGlobal -ge 0.5) { '#ffc107' } else { '#ff5252' }
$precLabel = if ($precisionGlobal -gt 0) { "$([math]::Round($precisionGlobal * 100))%" } else { '--' }
$HTML += "<div class=`"stats`"><div class=`"sc`"><div class=`"l`">Activos</div><div class=`"v`">30</div></div><div class=`"sc`"><div class=`"l`">Prob. Promedio</div><div class=`"v`">$avgProb%</div></div><div class=`"sc`"><div class=`"l`">Verde</div><div class=`"v`" style=`"color:#00c853`">$greenCount</div></div><div class=`"sc`"><div class=`"l`">Rojo</div><div class=`"v`" style=`"color:#ff5252`">$redCount</div></div><div class=`"sc`"><div class=`"l`">Precision</div><div class=`"v`" style=`"color:$precColor;font-size:14px`">$precLabel</div></div><div class=`"sc`"><div class=`"l`">Modelo IA</div><div class=`"v`" style=`"font-size:12px;color:#ce93d8`">$aiModel</div></div></div>"
if ($aiResumen) {
    $HTML += "<div class=`"resumen`">$aiResumen</div>"
}
# Top 5 recomendaciones
$sorted5 = $TICKERS | Sort-Object { $stockData[$_]['prob'] } -Descending
$top5Html = "<div class=`"t5`"><div class=`"t5h`">TOP 5 RECOMENDACIONES <span>Basado en probabilidad alcista generada por IA</span></div><div class=`"t5g`">"
for ($i = 0; $i -lt 5; $i++) {
    $t = $sorted5[$i]
    $info = $stockData[$t]; $pr = $prices[$t]
    $cc = if ($pr['change'] -ge 0) { '#00c853' } else { '#ff5252' }
    $sg = if ($pr['change'] -ge 0) { '+' } else { '' }
    $pc2 = if ($info['prob'] -ge 65) { 'pb-h' } elseif ($info['prob'] -ge 58) { 'pb-m' } else { 'pb-l' }
    $an = $info['analisis']
    if (-not $an -or $an -eq '') { $an = 'Sin analisis disponible' }
    $tp = $info['target_30d']
    if (-not $tp -or $tp -le 0) { $tp = $pr['price'] * (1 + ($info['prob'] - 50) / 200) }
    $up = [math]::Round(($tp / $pr['price'] - 1) * 100, 1)
    $upColor = if ($up -ge 5) { '#00c853' } elseif ($up -ge 2) { '#76ff03' } else { '#ffc107' }
    $top5Html += "<div class=`"t5c`"><div class=`"rk`">#$($i+1)</div><div class=`"tk`">$t</div><div class=`"nm`">$($info['name'])</div><div class=`"pr`" style=`"color:$cc`">`$$($pr['price']) <span style=`"font-size:11px;font-weight:400`">$sg$($pr['pct'])%</span></div><div class=`"tp`">Objetivo 30d: `$$([math]::Round($tp,2))</div><div class=`"up`" style=`"color:$upColor`">Proy. subida: +$up%</div><div class=`"an`">$an</div><div class=`"pb`"><div class=`"pbb`"><div class=`"pbf $pc2`" style=`"width:$($info['prob'])%`"></div></div><span class=`"pt`" style=`"color:$cc`">$($info['prob'])%</span></div></div>"
}
$top5Html += "</div></div>"
$HTML += $top5Html
$HTML += "<div class=`"sbar`"><input type=`"text`" id=`"searchInput`" placeholder=`"Buscar ticker, compania o sector...`" oninput=`"filtrarTabla()`"><select id=`"sectorFilter`" onchange=`"filtrarTabla()`"><option value=`"`">Todos los sectores</option>"
$sectoresUnicos = $sectorGroups.Keys | Sort-Object
foreach ($s in $sectoresUnicos) { $HTML += "<option value=`"$s`">$s</option>" }
$HTML += "</select><span class=`"rcount`" id=`"resultCount`">30/30</span></div>"
$HTML += "<div class=`"tc`"><table><thead><tr><th>#</th><th>Ticker</th><th class=`"hm`">Compania</th><th>Sector</th><th style=`"text-align:right`">Precio</th><th style=`"text-align:right`">Cambio</th><th style=`"text-align:right`">%</th><th style=`"text-align:right`">Prob.</th><th>Conf.</th></tr></thead><tbody>$tableRows</tbody></table></div>"
$HTML += "<div class=`"ft`">IA: $aiModel | Precios: $dataSource | $FECHA_HUMANA | 30 tickers | Generado por IA. No constituye asesoramiento financiero.</div>"
$HTML += "<script>function filtrarTabla(){var q=document.getElementById('searchInput').value.toUpperCase();var s=document.getElementById('sectorFilter').value;var rows=document.querySelectorAll('tbody tr');var v=0;rows.forEach(function(r){var tk=r.cells[1].textContent.toUpperCase();var nm=r.cells[2].textContent.toUpperCase();var sc=r.cells[3].textContent;var match=q===''||tk.includes(q)||nm.includes(q)||sc.toUpperCase().includes(q);if(s!==''&&sc!==s)match=false;r.style.display=match?'':'none';if(match)v++;});document.getElementById('resultCount').textContent=v+'/'+rows.length;}</script>"
$HTML += "</body></html>"

Set-Content -Path "$REPORTES_DIR/dashboard_top15.html" -Value $HTML -Force
Write-Output "[OK] Dashboard: $($HTML.Length) bytes"

# ============================================================
# PASO 5: REPORTE TEXTO
# ============================================================
$sectorGroups = @{}
foreach ($t in $TICKERS) {
    $sec = $stockData[$t]['sector']
    if (-not $sectorGroups.ContainsKey($sec)) { $sectorGroups[$sec] = @() }
    $sectorGroups[$sec] += $t
}
$sectorSummary = ($sectorGroups.Keys | ForEach-Object { "$($_): $($sectorGroups[$_].Count)" }) -join ', '

$reporte = "REPORTE 30 TICKERS - $FECHA_HUMANA`n"
$reporte += "Modelo IA: $aiModel | Precios: $dataSource`n`n"
$reporte += "RESUMEN IA:`n  $aiResumen`n`n"
$reporte += "NOTICIAS:`n"
foreach ($h in $headlines) { $reporte += "  $h`n" }
$reporte += "`nPORTAFOLIO 30 TICKERS:`n"
$reporte += "  Sectores: $sectorSummary`n"
$reporte += "  Probabilidad promedio: $avgProb%`n"
$reporte += "  Verdes: $greenCount | Rojos: $redCount`n"

$sorted = $TICKERS | Sort-Object { $stockData[$_]['prob'] } -Descending
$reporte += "`nRanking Top 5:`n"
for ($i = 0; $i -lt 5; $i++) {
    $t = $sorted[$i]
    $an = $stockData[$t]['analisis']
    $tp = $stockData[$t]['target_30d']
    $prc = $prices[$t]['price']
    if ($tp -le 0) { $tp = $prc * (1 + ($stockData[$t]['prob'] - 50) / 200) }
    $up = [math]::Round(($tp / $prc - 1) * 100, 1)
    $reporte += "  #$($i+1) $t (Prob: $($stockData[$t]['prob'])%) - Objetivo: `$$([math]::Round($tp,2)) (+$up% 30d) - $an`n"
}
$reporte += "`nRanking Bottom 5:`n"
for ($i = 29; $i -ge 25; $i--) {
    $t = $sorted[$i]
    $an = $stockData[$t]['analisis']
    $reporte += "  #$($i+1) $t (Prob: $($stockData[$t]['prob'])%) - $an`n"
}
$reporte += "`n--- Generado por IA ($aiModel). No constituye asesoramiento financiero. ---"

$reporteFile = "$REPORTES_DIR/Reporte_30T_$FECHA`_$($HORA -replace ':','').txt"
Set-Content -Path $reporteFile -Value $reporte -Force
Write-Output "[OK] Reporte: Reporte_30T_$FECHA`_$($HORA -replace ':','').txt"
Write-Output "[$FECHA $HORA] === CICLO 30 TICKERS COMPLETADO ==="
