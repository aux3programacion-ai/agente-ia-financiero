# ============================================================
# INSTALADOR DE TAREAS PROGRAMADAS
# Agente IA Auto-Evolutivo - Analisis Financiero
#
# EJECUTAR COMO ADMINISTRADOR (boton derecho > Ejecutar como Admin)
# ============================================================

Write-Host "=== INSTALADOR DE TAREAS PROGRAMADAS ===" -ForegroundColor Cyan
Write-Host "Agente IA - Ciclo de Analisis Financiero" -ForegroundColor Cyan
Write-Host ""

$scriptPath = "$env:USERPROFILE\Claro drive\AGENTE FINANCIERO\ciclo_automatizado.ps1"

if (-not (Test-Path $scriptPath)) {
    Write-Host "ERROR: No se encuentra el script en:" -ForegroundColor Red
    Write-Host $scriptPath -ForegroundColor Red
    exit 1
}

Write-Host "Script encontrado: $scriptPath" -ForegroundColor Green

# Verificar si se ejecuta como admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ADVERTENCIA: No se ejecuta como Administrador." -ForegroundColor Yellow
    Write-Host "Por favor, cierre esta ventana y ejecute como Administrador (boton derecho > Ejecutar como Administrador)." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Presione cualquier tecla para salir..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

try {
    # Eliminar tareas existentes si las hay
    schtasks /delete /tn "AgenteIA_Ciclo_Manana" /f 2>$null
    schtasks /delete /tn "AgenteIA_Ciclo_Tarde" /f 2>$null

    # Crear tarea de las 8:00 AM
    Write-Host "Creando tarea: AgenteIA_Ciclo_Manana (08:00)... " -NoNewline
    $result1 = schtasks /create /tn "AgenteIA_Ciclo_Manana" /tr "powershell -ExecutionPolicy Bypass -File `"$scriptPath`"" /sc daily /st 08:00 /f 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK" -ForegroundColor Green
    } else {
        Write-Host "ERROR" -ForegroundColor Red
        Write-Host $result1 -ForegroundColor Red
    }

    # Crear tarea de las 4:00 PM
    Write-Host "Creando tarea: AgenteIA_Ciclo_Tarde (16:00)... " -NoNewline
    $result2 = schtasks /create /tn "AgenteIA_Ciclo_Tarde" /tr "powershell -ExecutionPolicy Bypass -File `"$scriptPath`"" /sc daily /st 16:00 /f 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK" -ForegroundColor Green
    } else {
        Write-Host "ERROR" -ForegroundColor Red
        Write-Host $result2 -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "Verificando tareas registradas..." -ForegroundColor Cyan
    schtasks /query /tn "AgenteIA_Ciclo_Manana" /fo table 2>&1
    schtasks /query /tn "AgenteIA_Ciclo_Tarde" /fo table 2>&1

    Write-Host ""
    Write-Host "=== INSTALACION COMPLETADA ===" -ForegroundColor Green
    Write-Host "Las tareas se ejecutaran diariamente a las 08:00 y 16:00." -ForegroundColor Green
    Write-Host "Los resultados se guardan en: $env:USERPROFILE\Claro drive\AGENTE FINANCIERO\" -ForegroundColor Green
    Write-Host "Log de ejecucion: log_ejecucion.txt" -ForegroundColor Green

} catch {
    Write-Host "ERROR durante la instalacion: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Presione cualquier tecla para salir..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
