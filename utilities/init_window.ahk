if WinExist("Clash Royale")
{
    WinActivate
    WinMove 0, 0, 283, 518
}
else
{
    MsgBox "Clash Royale not found."
    ExitApp
}