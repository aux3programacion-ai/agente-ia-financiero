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
$BASE_DIR = if ($env:GITHUB_WORKSPACE) { $env:GITHUB_WORKSPACE } else { (Get-Location).Path }
$DATOS_DIR = "$BASE_DIR/Datos"
$REPORTES_DIR = "$BASE_DIR/Reportes"
New-Item -ItemType Directory -Path $DATOS_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $REPORTES_DIR -Force | Out-Null

Write-Output "[$FECHA $HORA] === CICLO 30 TICKERS ==="

# ============================================================
# PASO 1: CARGAR DATOS REALES + IA
# ============================================================
$TICKERS = @()
$stockData = @{}
$prices = @{}
$dataSource = 'simulada'
$aiModel = 'none'
$aiRegimen = ''
$aiResumen = ''
$aiFeedback = ''
$aiTitulares = @()
$aiSectores = @{}
$precisionGlobal = 0
$totalEvaluado = 0
$tickerMercados = @{}

# Cargar precios reales (load ALL prices first, regardless of TICKERS being empty)
$jsonPath = "$DATOS_DIR/precios_reales.json"
if (Test-Path $jsonPath) {
    try {
        $realData = Get-Content $jsonPath -Raw | ConvertFrom-Json
        $dataSource = $realData.fuente
        foreach ($tProp in $realData.precios.PSObject.Properties) {
            $t = $tProp.Name
            $pd = $tProp.Value
            if ($pd) {
                $prices[$t] = @{ 'price' = $pd.price; 'change' = $pd.change; 'pct' = $pd.pct }
            }
        }
    } catch { Write-Output "[!] Error cargando precios reales" }
}

# Cargar analisis IA (definir TICKERS desde analisis global)
$iaPath = "$DATOS_DIR/analisis_ia.json"
if (Test-Path $iaPath) {
    try {
        $iaData = Get-Content $iaPath -Raw | ConvertFrom-Json
        $aiModel = $iaData.modelo_usado
        $aiRegimen = $iaData.regimen_mercado
        $aiResumen = $iaData.resumen_mercado
        if ($iaData.titulares) { $aiTitulares = @($iaData.titulares) }
        if ($iaData.sectores) { $aiSectores = $iaData.sectores }
        if ($iaData.precision_global) { $precisionGlobal = $iaData.precision_global }
        if ($iaData.precision_ponderada) { $precisionGlobal = $iaData.precision_ponderada }
        if ($iaData.total_evaluado) { $totalEvaluado = $iaData.total_evaluado }
        if ($iaData.feedback_aprendizaje) { $aiResumen = $iaData.feedback_aprendizaje; $aiFeedback = $iaData.feedback_aprendizaje }
        # Use dynamic tickers from AI analysis (global universe)
        if ($iaData.tickers_analizados -and $iaData.tickers_analizados.Count -gt 0) {
            $TICKERS = @($iaData.tickers_analizados)
        } elseif ($iaData.probabilidades) {
            $TICKERS = @($iaData.probabilidades.PSObject.Properties.Name)
        }
        foreach ($t in $TICKERS) {
            if (-not $stockData.ContainsKey($t)) {
                $stockData[$t] = @{ 'prob' = 50; 'conf' = 50; 'sector' = 'Global'; 'name' = $t; 'analisis' = ''; 'target_30d' = 0; 'target_3m' = 0; 'target_6m' = 0; 'target_1y' = 0; 'precision_hist' = 0.5; 'mercado' = '' }
            }
            $iaInfo = $iaData.probabilidades.$t
            if ($iaInfo) {
                $stockData[$t]['prob'] = [int]$iaInfo.probabilidad
                $stockData[$t]['conf'] = [int]$iaInfo.confianza
                $stockData[$t]['analisis'] = [string]$iaInfo.analisis
                $stockData[$t]['target_30d'] = [double]($iaInfo.precio_objetivo_30d)
                $stockData[$t]['target_3m'] = [double]($iaInfo.precio_objetivo_3m)
                $stockData[$t]['target_6m'] = [double]($iaInfo.precio_objetivo_6m)
                $stockData[$t]['target_1y'] = [double]($iaInfo.precio_objetivo_1y)
                $stockData[$t]['mercado'] = [string]($iaInfo.mercado)
                $stockData[$t]['precision_hist'] = [double]($iaInfo.precision_historica)
            }
        }
        Write-Output "[OK] Analisis IA ($aiModel) cargado - $($TICKERS.Count) tickers globales"
    } catch { Write-Output "[!] Error cargando analisis IA" }
}

# Cargar screening global para mercado/origen (todos los mercados)
$screeningPath = "$DATOS_DIR/screening_global.json"
$screenPorMercado = @{}
if (Test-Path $screeningPath) {
    try {
        $scrData = Get-Content $screeningPath -Raw | ConvertFrom-Json
        $top50 = $scrData.top50
        foreach ($s in $top50) {
            $tickerMercados[$s.ticker] = $s.mercado
        }
        # Load full market-grouped data
        if ($scrData.por_mercado) {
            foreach ($m in $scrData.por_mercado.PSObject.Properties) {
                $screenPorMercado[$m.Name] = $m.Value
                foreach ($s in $m.Value) {
                    if (-not $tickerMercados.ContainsKey($s.ticker)) {
                        $tickerMercados[$s.ticker] = $s.mercado
                    }
                }
            }
        }
        if ($scrData.todos) {
            foreach ($s in $scrData.todos) {
                if (-not $tickerMercados.ContainsKey($s.ticker)) {
                    $tickerMercados[$s.ticker] = $s.mercado
                }
            }
        }
    } catch { Write-Output "[!] Error cargando screening global: $_" }
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

$sectorDefaults = @{}; $nameDefaults = @{}
foreach ($kv in $sectorMap.GetEnumerator()) { $sectorDefaults[$kv.Key] = $kv.Value }
foreach ($kv in $nameMap.GetEnumerator()) { $nameDefaults[$kv.Key] = $kv.Value }
# Expand TICKERS to include all screened tickers from all markets
$allScreenedTickers = @()
foreach ($m in $screenPorMercado.Keys) {
    foreach ($s in $screenPorMercado[$m]) {
        $allScreenedTickers += $s.ticker
    }
}
$TICKERS = @($TICKERS + $allScreenedTickers | Select-Object -Unique)

# Initialize stockData for any new tickers from screening
foreach ($t in $TICKERS) {
    if (-not $stockData.ContainsKey($t)) {
        $stockData[$t] = @{ 'prob' = 50; 'conf' = 50; 'sector' = 'Global'; 'name' = $t; 'analisis' = ''; 'target_30d' = 0; 'target_3m' = 0; 'target_6m' = 0; 'target_1y' = 0; 'precision_hist' = 0.5; 'mercado' = '' }
    }
}

foreach ($t in $TICKERS) {
    if ($sectorDefaults.ContainsKey($t)) { $stockData[$t]['sector'] = $sectorDefaults[$t] }
    if ($nameDefaults.ContainsKey($t)) { $stockData[$t]['name'] = $nameDefaults[$t] }
    if ($stockData[$t]['prob'] -eq 50 -and $probDefaults.ContainsKey($t)) {
        $stockData[$t]['prob'] = $probDefaults[$t]
        $stockData[$t]['conf'] = $confDefaults[$t]
    }
    if (-not $prices.ContainsKey($t)) {
        $base = if ($probDefaults.ContainsKey($t)) { $probDefaults[$t] * 3 + 50 } else { 200 }
        $var = $base * 0.015
        $live = [math]::Round($base + (Get-Random -Minimum -$var -Maximum $var), 2)
        $chg = [math]::Round($live - $base, 2)
        $prices[$t] = @{ 'price' = $live; 'change' = $chg; 'pct' = [math]::Round(($chg / $base) * 100, 2) }
    }
    # Set market label: try AI analysis, then screening, then suffix-based fallback
    if (-not $stockData[$t]['mercado'] -or $stockData[$t]['mercado'] -eq '') {
        $mFromScreen = $tickerMercados[$t]
        if ($mFromScreen) {
            $stockData[$t]['mercado'] = $mFromScreen
        } elseif ($t -match '\.MX$') {
            $stockData[$t]['mercado'] = 'MEXICO'
        } elseif ($t -match '\.DE$') {
            $stockData[$t]['mercado'] = 'EUROPA'
        } elseif ($t -match '\.L$') {
            $stockData[$t]['mercado'] = 'EUROPA'
        } elseif ($t -match '\.HK$|\.T$') {
            $stockData[$t]['mercado'] = 'ASIA'
        } else {
            $stockData[$t]['mercado'] = 'US'
        }
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
Add-Content -Path $globalPath -Value "[$FECHA] CICLO ${totalTickers}T INTRA $HORA - IA: $aiModel - $($headlines.Count) titulares" -Force
Write-Output "[OK] $totalTickers bitacoras actualizadas"

# Build sector groups (used in dashboard)
$sectorGroups = @{}
foreach ($t in $TICKERS) {
    $sec = $stockData[$t]['sector']
    if (-not $sectorGroups.ContainsKey($sec)) { $sectorGroups[$sec] = @() }
    $sectorGroups[$sec] += $t
}

# ============================================================
# PASO 4: DASHBOARD HTML
# ============================================================
$totalTickers = $TICKERS.Count
$probSum = 0; foreach ($t in $TICKERS) { $probSum += $stockData[$t]['prob'] }
$avgProb = if ($totalTickers -gt 0) { [math]::Round($probSum / $totalTickers) } else { 0 }
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
# Market display names and colors
$mercadoDisplay = @{
    'MEXICO' = 'MEXICO'; 'US' = 'ESTADOS UNIDOS'; 'ASIA' = 'ASIA'
    'EUROPA' = 'EUROPA'; 'GLOBAL' = 'GLOBAL'
}
$mercadoCSS = @{
    'MEXICO' = 'mmx'; 'US' = 'mus'; 'ASIA' = 'mas'; 'EUROPA' = 'meu'; 'GLOBAL' = 'mgl'
}
$mercadoColors = @{
    'MEXICO' = '#ffc107'; 'US' = '#00c853'; 'ASIA' = '#ff5252'; 'EUROPA' = '#64b5f6'; 'GLOBAL' = '#ce93d8'
}
$mercadoOrder = @('MEXICO', 'US', 'ASIA', 'EUROPA', 'GLOBAL')

# Generate market-grouped table rows
$marketTableRows = @{}
$marketCounts = @{}
$tickerListByMarket = @{}
foreach ($m in $mercadoOrder) {
    $marketTableRows[$m] = ""
    $marketCounts[$m] = 0
    $tickerListByMarket[$m] = @()
}
foreach ($t in $TICKERS) {
    $m = $stockData[$t]['mercado']
    if (-not $m) { $m = 'GLOBAL' }
    if (-not $marketCounts.ContainsKey($m)) { $marketCounts[$m] = 0; $marketTableRows[$m] = ""; $tickerListByMarket[$m] = @() }
    $tickerListByMarket[$m] += $t
}
foreach ($m in $mercadoOrder) {
    if ($marketCounts.ContainsKey($m) -and $tickerListByMarket[$m].Count -gt 0) {
        $sortedM = $tickerListByMarket[$m] | Sort-Object { $stockData[$_]['prob'] } -Descending
        $marketCounts[$m] = $sortedM.Count
        $rank = 1
        foreach ($t in $sortedM) {
            $info = $stockData[$t]; $pr = $prices[$t]
            if (-not $pr) { $pr = @{ 'price' = 0; 'change' = 0; 'pct' = 0 } }
            $cc = if ($pr['change'] -ge 0) { 'gn' } else { 'rd' }
            $sg = if ($pr['change'] -ge 0) { '+' } else { '' }
            $rc = if ($rank -eq 1) { 'r1' } elseif ($rank -eq 2) { 'r2' } elseif ($rank -eq 3) { 'r3' } else { '' }
            $sc = $sectorCSS[$info['sector']]
            $pc = if ($info['prob'] -ge 65) { 'pb-h' } elseif ($info['prob'] -ge 58) { 'pb-m' } else { 'pb-l' }
            $cf = if ($info['conf'] -ge 60) { 'cd-h' } elseif ($info['conf'] -ge 55) { 'cd-m' } else { 'cd-l' }
            $ph = $info['precision_hist']
            if ($ph -gt 0) { $phc = if ($ph -ge 0.7) { 'ph-h' } elseif ($ph -ge 0.5) { 'ph-m' } else { 'ph-l' } } else { $phc = 'ph-l'; $ph = 0 }
            $phLabel = if ($ph -gt 0) { "$([math]::Round($ph * 100))%" } else { '--' }
            $an = $info['analisis']
            $titleAttr = if ($an) { " title=`"$an`"" } else { "" }
            $marketTableRows[$m] += "<tr$titleAttr><td class=`"$rc`">$rank</td><td class=`"tk`">$t</td><td class=`"nm hm`">$($info['name'])</td><td><span class=`"sb $sc`">$($info['sector'])</span></td><td class=`"pr $cc`">`$$($pr['price'])</td><td class=`"ch $cc`">$sg`$$($pr['change'])</td><td class=`"ch $cc`">$sg$($pr['pct'])%</td><td><div class=`"pb`"><div class=`"pbb`"><div class=`"pbf $pc`" style=`"width:$($info['prob'])%`"></div></div><span class=`"pt`">$($info['prob'])%</span></div></td><td><div class=`"cc`"><span class=`"cd $cf`"></span>$($info['conf'])%</div></td><td class=`"ph $phc`">$phLabel</td></tr>"
            $rank++
        }
    }
}
# Build ticker bar items (all tickers)
$tickerItems = ""
foreach ($t in $TICKERS) {
    $pr = $prices[$t]
    if (-not $pr) { $pr = @{ 'price' = 0; 'change' = 0; 'pct' = 0 } }
    $sg = if ($pr['change'] -ge 0) { '+' } else { '' }
    $tc = if ($pr['change'] -ge 0) { 'cup' } else { 'cdn' }
    $tickerItems += "<span class=`"ti`"><span class=`"sy`">$t</span><span class=`"prc`">`$$($pr['price'])</span><span class=`"$tc`">$sg$($pr['pct'])%</span></span>"
}
# Build market summary strings
$marketSummary = ""
$marketStatsCards = ""
foreach ($m in $mercadoOrder) {
    if ($marketCounts.ContainsKey($m) -and $marketCounts[$m] -gt 0) {
        $mcolor = $mercadoColors[$m]
        $mdisplay = $mercadoDisplay[$m]
        if ($marketSummary) { $marketSummary += " | " }
        $marketSummary += "<span style='color:$mcolor;font-weight:700'>$mdisplay</span>: $($marketCounts[$m])"
        $marketStatsCards += "<div class='sc'><div class='l' style='color:$mcolor'>$mdisplay</div><div class='v' style='color:$mcolor'>$($marketCounts[$m])</div></div>"
    }
}

$CSS = "*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0b0e;color:#e8eaed}::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0a0b0e}::-webkit-scrollbar-thumb{background:#2a2d35}.hdr{background:linear-gradient(135deg,#0d0e12,#14161b);border-bottom:1px solid #1e2028;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}.logo{width:36px;height:36px;background:linear-gradient(135deg,#00c853,#00e676);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:16px;color:#0a0b0e}.hl{display:flex;align-items:center;gap:16px}.ht{font-size:16px;font-weight:700}.hs{font-size:11px;color:#6b7280}.badge{font-size:11px;font-weight:600;color:#00c853;background:rgba(0,200,83,0.08);padding:4px 12px;border-radius:20px}.stats{display:flex;gap:12px;padding:12px 24px;background:#0d0e12;border-bottom:1px solid #1e2028;flex-wrap:wrap}.sc{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:10px 16px;flex:1;min-width:100px}.sc .l{font-size:10px;text-transform:uppercase;color:#6b7280}.sc .v{font-size:16px;font-weight:700}.resumen{background:linear-gradient(135deg,#0d0e12,#14161b);border-bottom:1px solid #1e2028;padding:12px 24px;font-size:13px;color:#9ca3af}.tc{padding:24px;overflow-x:auto}table{width:100%;border-collapse:separate}th{font-size:10px;font-weight:600;text-transform:uppercase;color:#6b7280;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2028;background:#0a0b0e;position:sticky;top:0;z-index:2}td{padding:12px;font-size:13px;border-bottom:1px solid #15171d}tr:hover{background:rgba(0,200,83,0.03);cursor:default}.r1{color:#ffd700;font-weight:800}.r2{color:#c0c0c0;font-weight:800}.r3{color:#cd7f32;font-weight:800}.tk{font-weight:700}.nm{color:#9ca3af;font-size:12px}.pr{font-weight:700;text-align:right;font-variant-numeric:tabular-nums}.ch{font-weight:600;text-align:right}.gn{color:#00c853}.rd{color:#ff5252}.pb{display:flex;align-items:center;gap:8px}.pbb{flex:1;height:6px;background:#1e2028;border-radius:3px;overflow:hidden}.pbf{height:100%;border-radius:3px}.pb-h{background:linear-gradient(90deg,#00c853,#00e676)}.pb-m{background:linear-gradient(90deg,#ffc107,#ffd54f)}.pb-l{background:linear-gradient(90deg,#ff5252,#ff8a80)}.pt{font-weight:700;font-size:12px;min-width:36px;text-align:right}.cc{display:flex;align-items:center;gap:5px}.cd{width:5px;height:5px;border-radius:50%}.cd-h{background:#00c853}.cd-m{background:#ffc107}.cd-l{background:#ff5252}.ft{padding:16px 24px;border-top:1px solid #1e2028;font-size:10px;color:#4b5563;text-align:center}.tb{background:#0d0e12;border-bottom:1px solid #1e2028;padding:6px 0;overflow:hidden;white-space:nowrap}.tbi{display:inline-flex;gap:32px;animation:scroll 40s linear infinite}@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}.ti{display:inline-flex;align-items:center;gap:6px;font-size:12px}.ti .sy{font-weight:600;color:#9ca3af}.ti .prc{font-weight:700}.cup{color:#00c853}.cdn{color:#ff5252}.sb{font-size:10px;padding:2px 8px;border-radius:10px;display:inline-block;font-weight:500;background:#1e2028;color:#9ca3af}.sa{background:rgba(0,200,83,0.1);color:#00c853}.ss{background:rgba(33,150,243,0.1);color:#64b5f6}.sw{background:rgba(156,39,176,0.1);color:#ce93d8}.sc{background:rgba(255,152,0,0.1);color:#ffb74d}.st{background:rgba(0,188,212,0.1);color:#4dd0e1}.sm{background:rgba(233,30,99,0.1);color:#f48fb1}.scu{background:rgba(63,81,181,0.1);color:#9fa8da}.scc{background:rgba(63,81,181,0.1);color:#9fa8da}.sci{background:rgba(63,81,181,0.1);color:#9fa8da}.scl{background:rgba(156,39,176,0.1);color:#ce93d8}.sce{background:rgba(63,81,181,0.1);color:#9fa8da}.sph{background:rgba(0,150,136,0.1);color:#80cbc4}.seq{background:rgba(0,200,83,0.1);color:#00c853}.scd{background:rgba(63,81,181,0.1);color:#9fa8da}.sin{background:rgba(121,85,72,0.1);color:#a1887f}.smo{background:rgba(121,85,72,0.1);color:#a1887f}.sae{background:rgba(121,85,72,0.1);color:#a1887f}.sde{background:rgba(255,87,34,0.1);color:#ffab91}.sut{background:rgba(255,193,7,0.1);color:#ffe082}
.sbar{display:flex;gap:8px;padding:8px 24px;background:#0d0e12;border-bottom:1px solid #1e2028}.sbar input{flex:1;background:#14161b;border:1px solid #1e2028;border-radius:6px;padding:8px 12px;color:#e8eaed;font-size:13px;outline:none}.sbar input:focus{border-color:#00c853}.sbar input::placeholder{color:#4b5563}.sbar select{background:#14161b;border:1px solid #1e2028;border-radius:6px;padding:8px;color:#e8eaed;font-size:12px;outline:none;cursor:pointer}.sbar .rcount{color:#4b5563;font-size:11px;align-self:center;white-space:nowrap}
.ph{font-size:11px;text-align:right;font-weight:600}.ph-h{color:#00c853}.ph-m{color:#ffc107}.ph-l{color:#6b7280}
.fb{background:#0d0e12;border-bottom:1px solid #1e2028;padding:12px 24px}.fb summary{cursor:pointer;font-size:11px;font-weight:700;color:#00c853;display:flex;align-items:center;gap:6px}.fb pre{font-size:10px;color:#6b7280;line-height:1.5;margin-top:8px;white-space:pre-wrap;font-family:monospace}
.t5{padding:16px 24px;background:#0d0e12;border-bottom:1px solid #1e2028}.t5h{font-size:13px;font-weight:700;color:#00c853;margin-bottom:12px;display:flex;align-items:center;gap:8px}.t5h span{font-size:10px;color:#4b5563;font-weight:400}.t5g{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}.t5c{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:12px;position:relative;overflow:hidden}.t5c .rk{position:absolute;top:0;left:0;background:#00c853;color:#0a0b0e;font-size:10px;font-weight:800;padding:2px 8px;border-radius:0 0 6px 0}.t5c .tk{font-size:14px;font-weight:700;margin-top:6px}.t5c .nm{font-size:11px;color:#9ca3af;margin-bottom:4px}.t5c .pr{font-size:13px;font-weight:700}.t5c .tp{font-size:10px;color:#00c853;margin-top:2px;line-height:1.5}.t5c .an{font-size:10px;color:#6b7280;margin-top:6px;line-height:1.3}.t5c .pb{display:flex;align-items:center;gap:6px;margin-top:4px}.t5c .pb .pbb{flex:1;height:4px;background:#1e2028;border-radius:2px;overflow:hidden}.t5c .pb .pbf{height:100%;border-radius:2px}.t5c .pb .pt{font-size:11px;font-weight:700;min-width:30px;text-align:right}
.pf{padding:16px 24px;background:#0d0e12;border-bottom:1px solid #1e2028}.pfh{font-size:13px;font-weight:700;color:#e8eaed;margin-bottom:12px;display:flex;align-items:center;gap:8px}.pfh span{font-size:10px;color:#4b5563;font-weight:400}.pfi{display:flex;gap:8px;margin-bottom:12px}.pfi input{flex:1;background:#14161b;border:1px solid #1e2028;border-radius:6px;padding:8px 12px;color:#e8eaed;font-size:13px;outline:none;text-transform:uppercase}.pfi input:focus{border-color:#7c4dff}.pfi button{background:#7c4dff;border:none;border-radius:6px;padding:6px 16px;color:#fff;font-weight:700;font-size:18px;cursor:pointer}.pfi button:hover{background:#651fff}.pfl{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px}.pfc{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:12px;position:relative}.pfc .del{position:absolute;top:4px;right:6px;background:none;border:none;color:#6b7280;font-size:14px;cursor:pointer;padding:2px 4px}.pfc .del:hover{color:#ff5252}.pfc .tk{font-size:14px;font-weight:700}.pfc .nm{font-size:11px;color:#9ca3af;margin-bottom:2px}.pfc .pr{font-size:13px;font-weight:700}.pfc .scm{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:4px}.pfc .scm .sb{font-size:9px;padding:1px 6px;border-radius:4px;font-weight:600}.pfc .ns{margin-top:6px;padding-top:6px;border-top:1px solid #1e2028}.pfc .ns .nl{font-size:9px;color:#4b5563;text-transform:uppercase}.pfc .ns .nt{font-size:10px;color:#6b7280;line-height:1.3;margin-top:2px}.pfc .ns .nsm{font-size:9px;font-weight:600;margin-top:2px}.pfs{display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap}.pfs .pv{background:#14161b;border:1px solid #1e2028;border-radius:8px;padding:8px 14px;text-align:center}.pfs .pv .l{font-size:9px;text-transform:uppercase;color:#4b5563}.pfs .pv .v{font-size:15px;font-weight:700}
.mnb{display:flex;gap:6px;padding:8px 24px;background:#0d0e12;border-bottom:1px solid #1e2028;flex-wrap:wrap}.mnb-btn{background:transparent;border:1px solid #444;border-radius:6px;padding:6px 14px;color:#e8eaed;font-size:11px;font-weight:700;cursor:pointer;transition:all .15s}.mnb-btn:hover{background:rgba(255,255,255,0.05)}.mc{margin:0 24px 16px}.mc-hdr{display:flex;align-items:center;gap:12px;padding:12px 16px;background:#14161b;border-bottom:1px solid #1e2028;border-radius:8px 8px 0 0;margin-top:16px}.mc-hdr-txt{font-size:14px;font-weight:800;letter-spacing:0.5px}.mc-hdr-cnt{font-size:10px;color:#6b7280;font-weight:600}.ft{padding:16px 24px;text-align:center;font-size:10px;color:#4b5563;border-top:1px solid #1e2028}"

$HTML = "<!DOCTYPE html><html lang=`"es`"><head><meta charset=`"UTF-8`"><meta name=`"viewport`" content=`"width=device-width,initial-scale=1.0`"><title>Mercados Globales - IA Financiero $FECHA $HORA</title><style>$CSS</style></head><body>"
$HTML += "<header class=`"hdr`"><div class=`"hl`"><div class=`"logo`">AI</div><div><div class=`"ht`">Mercados Globales - $($TICKERS.Count) activos</div><div class=`"hs`">IA: $aiModel | Precios: $dataSource | $FECHA $HORA | $marketSummary</div></div></div><div class=`"badge`">$(if ($aiRegimen) { $aiRegimen.ToUpper() } else { 'GLOBAL' })</div></header>"
$HTML += "<div class=`"tb`"><div class=`"tbi`">$tickerItems $tickerItems</div></div>"
$HTML += "<div class=`"stats`">$marketStatsCards</div>"
if ($aiResumen) {
    $HTML += "<div class=`"resumen`">$aiResumen</div>"
}
# Feedback de aprendizaje
if ($aiFeedback) {
    $HTML += "<details class=`"fb`"><summary>APRENDIZAJE ACTIVO &rsaquo; Precision historica &amp; calibracion</summary><pre>$aiFeedback</pre></details>"
}
# Top picks by market
$topPicksHtml = ""
foreach ($m in $mercadoOrder) {
    if ($marketCounts.ContainsKey($m) -and $marketCounts[$m] -gt 0) {
        $sortedM = $tickerListByMarket[$m] | Sort-Object { $stockData[$_]['prob'] } -Descending
        $mcolor = $mercadoColors[$m]
        $mdisplay = $mercadoDisplay[$m]
        $n = [math]::Min(5, $sortedM.Count)
        $topPicksHtml += "<div class=`"t5`"><div class=`"t5h`" style=`"color:$mcolor`">TOP $n $mdisplay <span style=`"color:$mcolor!important`">Mejores probabilidades del mercado</span></div><div class=`"t5g`">"
        for ($i = 0; $i -lt $n; $i++) {
            $t = $sortedM[$i]
            $info = $stockData[$t]; $pr = $prices[$t]
            if (-not $pr) { $pr = @{ 'price' = 0; 'change' = 0; 'pct' = 0 } }
            $cc = if ($pr['change'] -ge 0) { '#00c853' } else { '#ff5252' }
            $sg = if ($pr['change'] -ge 0) { '+' } else { '' }
            $pc2 = if ($info['prob'] -ge 65) { 'pb-h' } elseif ($info['prob'] -ge 58) { 'pb-m' } else { 'pb-l' }
            $an = $info['analisis']
            if (-not $an -or $an -eq '') { $an = 'Sin analisis disponible' }
            $tp30 = $info['target_30d']
            $prc = $pr['price']
            if ($prc -le 0) { $prc = 100 }
            if (-not $tp30 -or $tp30 -le 0) { $tp30 = $prc * (1 + ($info['prob'] - 50) / 200) }
            $up30 = [math]::Round(($tp30 / $prc - 1) * 100, 1)
            $ph = $info['precision_hist']
            $phLabelN = if ($ph -gt 0) { "Prec: $([math]::Round($ph * 100))%" } else { '' }
            $phDivN = if ($phLabelN) { "<div style=`"font-size:9px;color:#6b7280;margin-top:4px`">$phLabelN</div>" } else { '' }
            $topPicksHtml += "<div class=`"t5c`"><div class=`"rk`">#$($i+1)</div><div class=`"tk`">$t</div><div class=`"nm`">$($info['name'])</div><div class=`"pr`" style=`"color:$cc`">`$$($prc) <span style=`"font-size:11px;font-weight:400`">$sg$($pr['pct'])%</span></div><div class=`"tp`">30d: `$$([math]::Round($tp30,2)) (+$up30%)</div><div class=`"an`">$an</div><div class=`"pb`"><div class=`"pbb`"><div class=`"pbf $pc2`" style=`"width:$($info['prob'])%`"></div></div><span class=`"pt`" style=`"color:$cc`">$($info['prob'])%</span></div>$phDivN</div>"
        }
        $topPicksHtml += "</div></div>"
    }
}
$HTML += $topPicksHtml

# Build embedded data for portfolio JS
$pfData = @{}
foreach ($t in $TICKERS) {
    $pr = $prices[$t]; $info = $stockData[$t]
    $tp = $info['target_30d']
    if (-not $tp -or $tp -le 0) { $tp = $pr['price'] * (1 + ($info['prob'] - 50) / 200) }
    $prc = $pr['price']
    $tp30f = $info['target_30d']; $tp3mf = $info['target_3m']; $tp6mf = $info['target_6m']; $tp1yf = $info['target_1y']
    if (-not $tp30f -or $tp30f -le 0) { $tp30f = $prc * (1 + ($info['prob'] - 50) / 200) }
    if (-not $tp3mf -or $tp3mf -le 0) { $tp3mf = $prc * (1 + ($info['prob'] - 50) / 150) }
    if (-not $tp6mf -or $tp6mf -le 0) { $tp6mf = $prc * (1 + ($info['prob'] - 50) / 100) }
    if (-not $tp1yf -or $tp1yf -le 0) { $tp1yf = $prc * (1 + ($info['prob'] - 50) / 80) }
    $pfData[$t] = @{p=$pr['price'];ch=$pr['change'];pc=$pr['pct'];pr=$info['prob'];cf=$info['conf'];tg=[math]::Round($tp30f,2);tg3m=[math]::Round($tp3mf,2);tg6m=[math]::Round($tp6mf,2);tg1y=[math]::Round($tp1yf,2);an=$info['analisis'];nm=$info['name'];sc=$info['sector'];ph=$info['precision_hist'];mc=$info['mercado']}
}
$pfJson = $pfData | ConvertTo-Json -Compress

$newsForJs = @{}
$newsPath = "$DATOS_DIR/noticias_recientes.json"
if (Test-Path $newsPath) {
    try {
        $newsRaw = Get-Content $newsPath -Raw | ConvertFrom-Json
        foreach ($t in $TICKERS) {
            $nd = $newsRaw.por_ticker.$t
            if ($nd -and $nd.noticias) {
                $notis = @()
                foreach ($n in $nd.noticias) {
                    $st = if ($nd.sentimiento -and $nd.sentimiento.sentimiento) { $nd.sentimiento.sentimiento } else { 'neutral' }
                    $ss = if ($nd.sentimiento -and $nd.sentimiento.score -ne '') { $nd.sentimiento.score } else { 0 }
                    $notis += @{t=$n.titulo;s=$st;sc=$ss}
                }
                if ($notis.Count -gt 0) { $newsForJs[$t] = $notis[0] }
            }
        }
    } catch {}
}
$newsJson = $newsForJs | ConvertTo-Json -Compress

# Read portfolio from repo file (sincronizado entre dispositivos)
$pfFileArray = @()
$pfFilePath = "$DATOS_DIR/portafolio_usuario.json"
if (Test-Path $pfFilePath) {
    try {
        $pfFileArray = Get-Content $pfFilePath -Raw | ConvertFrom-Json
        if (-not $pfFileArray -or $pfFileArray.Count -eq 0) { $pfFileArray = @() }
    } catch { $pfFileArray = @() }
}
$pfFileJson = $pfFileArray | ConvertTo-Json -Compress
if (-not $pfFileJson) { $pfFileJson = '[]' }

# Portfolio section HTML
$HTML += "<div class=`"pf`"><div class=`"pfh`">MI PORTAFOLIO <span>Se sincroniza automaticamente en todos tus dispositivos via GitHub - edita <b>Datos/portafolio_usuario.json</b> en el repo</span></div>"
$HTML += "<div class=`"pfi`"><input type=`"text`" id=`"pfInput`" placeholder=`"Ej: NVDA, AAPL, MSFT`" onkeydown=`"if(event.key==='Enter')agregarAlPortafolio()`"><button onclick=`"agregarAlPortafolio()`">+</button></div>"
$HTML += "<div class=`"pfs`" id=`"pfStats`"></div>"
$HTML += "<div id=`"pfStatus`"></div>"
$HTML += "<div class=`"pfl`" id=`"pfList`"><div style=`"color:#4b5563;font-size:12px;grid-column:1/-1;text-align:center;padding:20px`">Agrega tickers arriba para seguir su probabilidad en tiempo real</div></div></div>"
# Build market navigation tabs + search bar
$mercadoNav = "<div class=`"mnb`"><button class=`"mnb-btn`" style=`"border-color:#00c853;color:#00c853`" onclick=`"mostrarTodos()`">TODOS</button>"
foreach ($m in $mercadoOrder) {
    if ($marketCounts.ContainsKey($m) -and $marketCounts[$m] -gt 0) {
        $mcolor = $mercadoColors[$m]
        $mdisplay = $mercadoDisplay[$m]
        $mercadoNav += "<button class=`"mnb-btn`" data-market=`"$m`" style=`"border-color:$mcolor;color:$mcolor`" onclick=`"showMercado('$m')`">$mdisplay ($($marketCounts[$m]))</button>"
    }
}
$mercadoNav += "</div>"

# Build search bar with market filter
$HTML += "<div class=`"sbar`"><input type=`"text`" id=`"searchInput`" placeholder=`"Buscar ticker, compania o sector...`" oninput=`"filtrarMercados()`"><select id=`"sectorFilter`" onchange=`"filtrarMercados()`"><option value=`"`">Todos los sectores</option>"
$sectoresUnicos = $sectorGroups.Keys | Sort-Object
foreach ($s in $sectoresUnicos) { $HTML += "<option value=`"$s`">$s</option>" }
$HTML += "</select><select id=`"mercadoFilter`" onchange=`"filtrarMercados()`"><option value=`"`">Todos los mercados</option>"
foreach ($m in $mercadoOrder) { if ($marketCounts.ContainsKey($m) -and $marketCounts[$m] -gt 0) { $HTML += "<option value=`"$m`">$($mercadoDisplay[$m])</option>" } }
$HTML += "</select><span class=`"rcount`" id=`"resultCount`">$($TICKERS.Count)/$($TICKERS.Count)</span></div>"
$HTML += $mercadoNav

# Build market-grouped tables
foreach ($m in $mercadoOrder) {
    if ($marketCounts.ContainsKey($m) -and $marketCounts[$m] -gt 0) {
        $mcolor = $mercadoColors[$m]
        $mdisplay = $mercadoDisplay[$m]
        $rows = $marketTableRows[$m]
        $HTML += "<div class=`"mc`" id=`"mc-$m`"><div class=`"mc-hdr`" style=`"border-left:4px solid $mcolor`"><span class=`"mc-hdr-txt`" style=`"color:$mcolor`">$mdisplay</span><span class=`"mc-hdr-cnt`">$($marketCounts[$m]) activos</span></div>"
        $HTML += "<div class=`"tc`"><table><thead><tr><th>#</th><th>Ticker</th><th class=`"hm`">Compania</th><th>Sector</th><th style=`"text-align:right`">Precio</th><th style=`"text-align:right`">Cambio</th><th style=`"text-align:right`">%</th><th style=`"text-align:right`">Prob.</th><th>Conf.</th><th style=`"text-align:right`">Prec.H</th></tr></thead><tbody>$rows</tbody></table></div></div>"
    }
}
$HTML += "<div class=`"ft`">IA: $aiModel | Precios: $dataSource | $FECHA_HUMANA | $($TICKERS.Count) tickers globales | Generado por IA. No constituye asesoramiento financiero.</div>"
$HTML += "<script>function showMercado(m){document.querySelectorAll('.mc').forEach(function(e){e.style.display=e.id==='mc-'+m?'block':'none'});document.querySelectorAll('.mnb-btn').forEach(function(b){b.style.opacity=b.getAttribute('data-market')===m?'1':'0.4'});}function mostrarTodos(){document.querySelectorAll('.mc').forEach(function(e){e.style.display='block'});document.querySelectorAll('.mnb-btn').forEach(function(b){b.style.opacity='1'});}function filtrarMercados(){var q=document.getElementById('searchInput').value.toUpperCase();var s=document.getElementById('sectorFilter').value;var mf=document.getElementById('mercadoFilter').value;var total=0;var vis=0;document.querySelectorAll('.mc').forEach(function(mc){var mercadoId=mc.id.replace('mc-','');if(mf!==''&&mercadoId!==mf){mc.style.display='none';return}mc.style.display='block';var rows=mc.querySelectorAll('tbody tr');var cv=0;rows.forEach(function(r){var tk=r.cells[1].textContent.toUpperCase();var nm=r.cells[2].textContent.toUpperCase();var sc=r.cells[3].textContent;var match=q===''||tk.includes(q)||nm.includes(q)||sc.toUpperCase().includes(q);if(s!==''&&sc!==s)match=false;r.style.display=match?'':'none';if(match)cv++});if(cv===0)mc.style.display='none';vis+=cv;total+=rows.length});document.getElementById('resultCount').textContent=vis+'/'+total;}</script>"
$LIVE_SCRIPT = @'
<script>
(function(){var PS=['https://api.allorigins.win/raw?url=','https://corsproxy.io/?url=','https://api.codetabs.com/v1/proxy?quest=','https://thingproxy.freeboard.io/fetch/'],B=10;
function apply(t,p,ch,pc){if(PD[t]){PD[t].p=Math.round(p*100)/100;PD[t].ch=Math.round(ch*100)/100;PD[t].pc=Math.round(pc*100)/100}
var gn=ch>=0,sg=gn?'+':'',cl=gn?'gn':'rd';
var rows=document.querySelectorAll('tbody tr');
for(var i=0;i<rows.length;i++){var tk=rows[i].querySelector('.tk');
if(tk&&tk.textContent.trim()===t&&rows[i].cells.length>=7){
rows[i].cells[4].textContent='$'+p.toFixed(2);rows[i].cells[4].className='pr '+cl;
rows[i].cells[5].textContent=sg+'$'+ch.toFixed(2);rows[i].cells[5].className='ch '+cl;
rows[i].cells[6].textContent=sg+pc.toFixed(2)+'%';rows[i].cells[6].className='ch '+cl;}}
var items=document.querySelectorAll('.ti');
for(var i=0;i<items.length;i++){var sy=items[i].querySelector('.sy');
if(sy&&sy.textContent.trim()===t){
var pr=items[i].querySelector('.prc'),pt=items[i].querySelector('.cup,.cdn');
if(pr)pr.textContent='$'+p.toFixed(2);
if(pt){pt.textContent=sg+pc.toFixed(2)+'%';pt.className=pc>=0?'cup':'cdn';}}}}
function fetchBatch(b,i){if(i>=PS.length)return;
var u=PS[i]+encodeURIComponent('https://query1.finance.yahoo.com/v8/finance/chart/'+b.join(',')+'?interval=1d&range=5d');
fetch(u).then(function(r){if(!r.ok)throw Error();return r.json()}).then(function(d){
if(!d.chart||!d.chart.result)throw Error();
d.chart.result.forEach(function(rr){try{var m=rr.meta,s=m.symbol,p=m.regularMarketPrice;
if(!p||!s)return;
var qc=rr.indicators.quote[0].close.filter(function(v){return v!==null});
var pp=qc.length>0?qc[qc.length-1]:p,ch=p-pp,pc=pp>0?(ch/pp*100):0;
apply(s,p,ch,pc);}catch(e){}});}).catch(function(){fetchBatch(b,i+1)});}
function up(){var ts=Object.keys(PD),batches=[];
for(var i=0;i<ts.length;i+=B)batches.push(ts.slice(i,i+B));
batches.forEach(function(b){fetchBatch(b,0)});}
window.addEventListener('load',function(){setTimeout(up,1000)});
setInterval(up,45000);})();
</script>
'@
$HTML += "<script src=`"./data.js`"></script>"
$HTML += "<script src=`"./portfolio.js`"></script>"
$HTML += $LIVE_SCRIPT
$HTML += "</body></html>"

$dataJs = "var PD=$pfJson;var ND=$newsJson;var PF=$pfFileJson;"
Set-Content -Path "$REPORTES_DIR/data.js" -Value $dataJs -Force
Set-Content -Path "$REPORTES_DIR/dashboard_top15.html" -Value $HTML -Force
Copy-Item -Path "$BASE_DIR/portfolio.js" -Destination "$REPORTES_DIR/portfolio.js" -Force
Write-Output "[OK] Dashboard: $($HTML.Length) bytes | data.js: $($dataJs.Length) bytes"

# ============================================================
# PASO 5: REPORTE TEXTO
# ============================================================
$sectorSummary = ($sectorGroups.Keys | ForEach-Object { "$($_): $($sectorGroups[$_].Count)" }) -join ', '

$reporte = "REPORTE $totalTickers TICKERS - $FECHA_HUMANA`n"
$reporte += "Modelo IA: $aiModel | Precios: $dataSource`n`n"
$reporte += "RESUMEN IA:`n  $aiResumen`n`n"
$reporte += "NOTICIAS:`n"
foreach ($h in $headlines) { $reporte += "  $h`n" }
$reporte += "`nPORTAFOLIO $totalTickers TICKERS:`n"
$reporte += "  Sectores: $sectorSummary`n"
$reporte += "  Probabilidad promedio: $avgProb%`n"
$reporte += "  Verdes: $greenCount | Rojos: $redCount`n"

$sorted = $TICKERS | Sort-Object { $stockData[$_]['prob'] } -Descending
$nTop = [math]::Min(5, $totalTickers)
$reporte += "`nRanking Top ${nTop}:`n"
for ($i = 0; $i -lt $nTop; $i++) {
    $t = $sorted[$i]
    $an = $stockData[$t]['analisis']
    $tp = $stockData[$t]['target_30d']
    $prc = $prices[$t]['price']
    if ($tp -le 0) { $tp = $prc * (1 + ($stockData[$t]['prob'] - 50) / 200) }
    $up = [math]::Round(($tp / $prc - 1) * 100, 1)
    $reporte += "  #$($i+1) $t (Prob: $($stockData[$t]['prob'])%) - Objetivo: `$$([math]::Round($tp,2)) (+$up% 30d) - $an`n"
}
$nBottom = [math]::Min(5, $totalTickers)
$reporte += "`nRanking Bottom ${nBottom}:`n"
for ($i = $totalTickers - 1; $i -ge ($totalTickers - $nBottom); $i--) {
    $t = $sorted[$i]
    $an = $stockData[$t]['analisis']
    $reporte += "  #$($i+1) $t (Prob: $($stockData[$t]['prob'])%) - $an`n"
}
$reporte += "`n--- Generado por IA ($aiModel). No constituye asesoramiento financiero. ---"

$reporteFile = "$REPORTES_DIR/Reporte_${totalTickers}T_$FECHA`_$($HORA -replace ':','').txt"
Set-Content -Path $reporteFile -Value $reporte -Force
Write-Output "[OK] Reporte: Reporte_30T_$FECHA`_$($HORA -replace ':','').txt"
Write-Output "[$FECHA $HORA] === CICLO 30 TICKERS COMPLETADO ==="
