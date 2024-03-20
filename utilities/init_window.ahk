if WinExist("Clash Royale")
{
    WinActivate
    WinMove 0, 0, 295, 535
}
else
{
    MsgBox "Clash Royale not found."
    ExitApp
}