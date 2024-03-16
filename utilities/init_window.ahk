if WinExist("Clash Royale")
{
    WinActivate
    WinMove 0, 0, 436, 792
}
else
{
    MsgBox "Clash Royale not found."
    ExitApp
}