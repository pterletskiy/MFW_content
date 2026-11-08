# GitHub logout helper for pterletskiy
# Revokes gh CLI token for pterletskiy only (leaves other accounts alone),
# and clears git identity from global config.

$ErrorActionPreference = "Continue"

# Force UTF-8 in the console so gh output renders correctly.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$GitUser = "pterletskiy"

Write-Host "=== GitHub Logout ===" -ForegroundColor Cyan

# 1. Log out ONLY the pterletskiy account (leave any other accounts intact)
Write-Host "`n[1/3] Logging out of GitHub CLI for account '$GitUser'..." -ForegroundColor Yellow
gh auth logout --hostname github.com --user $GitUser 2>&1 | Out-Host
if ($LASTEXITCODE -eq 0) {
    Write-Host "      gh CLI session for '$GitUser' ended."
} else {
    Write-Host "      Could not end gh session for '$GitUser'. It may already be logged out."
}

# 2. Clear git identity from global config
Write-Host "`n[2/3] Clearing git identity..." -ForegroundColor Yellow
git config --global --unset-all user.name  2>$null
git config --global --unset-all user.email 2>$null
$remainingName  = git config --global user.name  2>$null
$remainingEmail = git config --global user.email 2>$null
if ([string]::IsNullOrEmpty($remainingName) -and [string]::IsNullOrEmpty($remainingEmail)) {
    Write-Host "      git identity cleared."
} else {
    Write-Host "      Warning: some identity fields still present." -ForegroundColor Red
    Write-Host "        user.name  = $remainingName"
    Write-Host "        user.email = $remainingEmail"
}

# 3. Final verification (list all remaining accounts)
Write-Host "`n[3/3] Verifying logout..." -ForegroundColor Yellow
$finalStatus = gh auth status 2>&1 | Out-String
if ($finalStatus -match "Logged in to github\.com account $GitUser") {
    Write-Host "      Warning: '$GitUser' still appears in gh auth status." -ForegroundColor Red
    Write-Host $finalStatus
} else {
    Write-Host "      '$GitUser' successfully removed from gh CLI."
    Write-Host "`n      Remaining gh accounts on this machine:"
    Write-Host $finalStatus
}

Write-Host "`nLogout complete. Your '$GitUser' account is no longer accessible from this session." -ForegroundColor Green
Write-Host "Other GitHub accounts on this machine (if any) were not touched." -ForegroundColor Green
Write-Host "Run github_login.ps1 next time you need to work on the repos.`n" -ForegroundColor Green