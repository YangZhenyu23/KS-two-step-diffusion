# sync.ps1 — 提交本地改动并推送到 GitHub
#
# 用法：
#   .\scripts\sync.ps1 -Message "补充 sigma=10 的云图"
#   .\scripts\sync.ps1 -Message "修复 analyze.py 路径" -NoPush
#   .\scripts\sync.ps1 -DryRun
#
# 行为：
#   1. 检查仓库状态；无改动则直接退出
#   2. 跑一次 .gitignore 校验，防止误提交大文件
#   3. git add -A → commit（提交信息自动带 UTC+8 时间戳）→ push
#
# 若提交信息未以「新增:/变更:/修复:/移除:/复现:/文档:」开头，脚本会补上前缀「变更:」。

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Message = "",

    [switch]$NoPush,
    [switch]$DryRun,

    [string]$Remote = "origin",
    [string]$Branch = ""
)

$ErrorActionPreference = "Stop"
$env:GIT_PAGER = "cat"

function Write-Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "    $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "    $text" -ForegroundColor Yellow }
function Write-Err($text)  { Write-Host "    $text" -ForegroundColor Red }

# ---- 定位仓库根目录 ----
try {
    $repoRoot = (& git rev-parse --show-toplevel 2>$null)
} catch {
    Write-Err "当前目录不是 git 仓库。请先在项目根目录执行 git init。"
    exit 1
}
if (-not $repoRoot) {
    Write-Err "当前目录不是 git 仓库。"
    exit 1
}
Set-Location $repoRoot
Write-Step "仓库：$repoRoot"

# ---- 当前分支 ----
if (-not $Branch) {
    $Branch = (& git rev-parse --abbrev-ref HEAD).Trim()
}
Write-Step "分支：$Branch"

# ---- 检查改动 ----
$status = & git status --porcelain
if (-not $status) {
    Write-Ok "工作区干净，无需提交。"
    if (-not $NoPush) {
        $ahead = (& git rev-list --count "$Remote/$Branch..HEAD" 2>$null)
        if ($ahead -and [int]$ahead -gt 0) {
            Write-Warn "但有 $ahead 个本地提交尚未推送。"
            if (-not $DryRun) {
                & git push $Remote $Branch
                if ($LASTEXITCODE -eq 0) { Write-Ok "已推送 $ahead 个提交。" }
                else { Write-Err "推送失败。"; exit 1 }
            }
        }
    }
    exit 0
}

Write-Step "检测到以下改动："
$status | ForEach-Object { Write-Host "    $_" }

# ---- 大文件防护：检查暂存区里是否有被 .gitignore 排除却仍被跟踪的文件 ----
Write-Step "检查大文件（>20 MB）"
$big = @()
if (Test-Path ".git") {
    $tracked = & git ls-files
    foreach ($f in $tracked) {
        if (Test-Path -LiteralPath $f) {
            $len = (Get-Item -LiteralPath $f).Length
            if ($len -gt 20MB) {
                $big += [pscustomobject]@{ File = $f; MB = [math]::Round($len / 1MB, 1) }
            }
        }
    }
}
if ($big.Count -gt 0) {
    Write-Warn "发现超过 20 MB 的已跟踪文件："
    $big | ForEach-Object { Write-Host ("      {0}  ({1} MB)" -f $_.File, $_.MB) -ForegroundColor Yellow }
    Write-Warn "如果这些不该入库，请先 git rm --cached <file> 并确认 .gitignore 已覆盖。"
} else {
    Write-Ok "无超大文件。"
}

# ---- 组织提交信息 ----
if (-not $Message) {
    if ($DryRun) {
        $Message = "<提交说明>"
    } else {
        $Message = Read-Host "请输入提交说明（中文，例如：补充 sigma=10 的云图）"
        if (-not $Message) {
            Write-Err "提交说明不能为空。"
            exit 1
        }
    }
}
$prefixes = @("新增:", "变更:", "修复:", "移除:", "复现:", "文档:", "新增：", "变更：", "修复：", "移除：", "复现：", "文档：")
$hasPrefix = $false
foreach ($p in $prefixes) {
    if ($Message.StartsWith($p)) { $hasPrefix = $true; break }
}
if (-not $hasPrefix) {
    $Message = "变更: $Message"
    Write-Warn "已补类型前缀 -> $Message"
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message`n`n同步时间: $stamp (UTC+8)"

# ---- 执行 ----
if ($DryRun) {
    Write-Step "[DryRun] 将执行："
    Write-Host "    git add -A"
    Write-Host "    git commit -m `"$fullMessage`""
    if (-not $NoPush) { Write-Host "    git push $Remote $Branch" }
    exit 0
}

Write-Step "git add -A"
& git add -A
if ($LASTEXITCODE -ne 0) { Write-Err "git add 失败。"; exit 1 }

$staged = & git diff --cached --name-only
if (-not $staged) {
    Write-Ok "暂存区为空（改动可能都被 .gitignore 排除了）。"
    exit 0
}
Write-Ok "已暂存 $($staged.Count) 个文件。"

Write-Step "git commit"
& git commit -m $fullMessage
if ($LASTEXITCODE -ne 0) { Write-Err "git commit 失败。"; exit 1 }
Write-Ok "提交完成。"

if ($NoPush) {
    Write-Warn "按 -NoPush 要求，未推送。手动推送：git push $Remote $Branch"
    exit 0
}

Write-Step "git push $Remote $Branch"
& git push $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Err "推送失败。常见原因：远端有新提交（先 git pull --rebase）、凭据过期、网络问题。"
    exit 1
}
Write-Ok "已同步到 GitHub。"

Write-Host ""
Write-Host "别忘了在 CHANGELOG.md 的 [Unreleased] 下登记本次改动。" -ForegroundColor Magenta
