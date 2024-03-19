; script to move card to coordinate
x := A_Args[1]
y := A_Args[2]
cardx := A_Args[3]
cardy := A_Args[4]

MouseMove cardx, cardy
Click

MouseMove x, y
Click