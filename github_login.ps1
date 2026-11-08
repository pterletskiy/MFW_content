# GitHub login helper for pterletskiy
# Sets git identity and authenticates GitHub CLI for this session.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$GitUser  = "pterletskiy"
$GitEmail = "petr.terletskiy@gmail.com"

Write-Host "=== GitHub Login ===" -ForegroundColor Cyan

# 1. Set git identity globally
Write-Host "`n[1/3] Setting git identity..." -ForegroundColor Yellow
git config --global --replace-all user.name  $GitUser
git config --global --replace-all user.email $GitEmail

$currentName  = git config --global user.name
$currentEmail = git config --global user.email
Write-Host "      user.name  = $currentName"
Write-Host "      user.email = $currentEmail"

# 2. Check GitHub CLI auth status
Write-Host "`n[2/3] Checking GitHub CLI auth..." -ForegroundColor Yellow
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -eq 0 -and $authStatus -match "Logged in to github.com account $GitUser") {
    Write-Host "      Already authenticated as $GitUser."
} else {
    Write-Host "      Not authenticated. Launching browser login..."
    gh auth login --hostname github.com --git-protocol https --web
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nAuthentication failed. Aborting." -ForegroundColor Red
        exit 1
    }
}

# 3. Confirm final state
Write-Host "`n[3/3] Final verification..." -ForegroundColor Yellow
gh auth status

Write-Host "`nLogin complete. You are set up as $GitUser on this machine." -ForegroundColor Green
Write-Host "Remember to run github_logout.ps1 when you finish your session.`n" -ForegroundColor Green