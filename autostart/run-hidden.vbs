' Launches a command with no console window.
' Used by the Startup-folder entries so logon does not flash two black windows.
Set sh = CreateObject("WScript.Shell")
If WScript.Arguments.Count > 0 Then
  sh.Run """" & WScript.Arguments(0) & """", 0, False
End If
