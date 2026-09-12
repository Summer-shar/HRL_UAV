param(
  [Parameter(Mandatory=$true)][string]$InputFile,
  [Parameter(Mandatory=$true)][string]$PdfFile,
  [string]$DocxCopy
)
$ErrorActionPreference = 'Stop'
$wordApp = $null
$wordDoc = $null
try {
  $wordApp = New-Object -ComObject Word.Application
  $wordApp.Visible = $false
  $wordApp.DisplayAlerts = 0
  $wordApp.AutomationSecurity = 3
  $wordDoc = $wordApp.Documents.Open($InputFile, $false, $true, $false)
  if ($DocxCopy) { $wordDoc.SaveAs2($DocxCopy, 16) }
  $wordDoc.Repaginate()
  $wordDoc.ExportAsFixedFormat($PdfFile, 17)
  Write-Output "Word export complete; pages: $($wordDoc.ComputeStatistics(2))"
  if ($DocxCopy) { Write-Output $DocxCopy }
  Write-Output $PdfFile
} finally {
  if ($null -ne $wordDoc) { $wordDoc.Close(0) }
  if ($null -ne $wordApp) { $wordApp.Quit(0) }
  if ($null -ne $wordDoc) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordDoc) }
  if ($null -ne $wordApp) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) }
}
