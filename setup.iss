[Setup]
AppName=CodeInsight AI代码分析平台
AppVersion=1.0.0
AppPublisher=CodeInsight Team
DefaultDirName={pf}\CodeInsight
DefaultGroupName=CodeInsight
OutputDir=Output
OutputBaseFilename=CodeInsight-Setup
SetupIconFile=assets\icon.ico
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs
Source: "assets\install\precheck.ps1"; DestDir: "{app}\install"; Flags: ignoreversion
Source: "assets\install\post_install.ps1"; DestDir: "{app}\install"; Flags: ignoreversion
Source: "assets\install\create-shortcut.ps1"; DestDir: "{app}\install"; Flags: ignoreversion
Source: "assets\install\verify-shortcut.ps1"; DestDir: "{app}\install"; Flags: ignoreversion
Source: "assets\install\preuninstall.ps1"; DestDir: "{app}\install"; Flags: ignoreversion
Source: "assets\install\diagnostics.ps1"; DestDir: "{app}\install"; Flags: ignoreversion
Source: "assets\install\fix.ps1"; DestDir: "{app}\install"; Flags: ignoreversion

[Icons]
Name: "{group}\CodeInsight"; Filename: "{app}\CodeInsight.exe"; Tasks: startmenuicon; IconFilename: "{app}\assets\icon.ico"
Name: "{commondesktop}\CodeInsight"; Filename: "{app}\CodeInsight.exe"; Tasks: desktopicon; IconFilename: "{app}\assets\icon.ico"

[Run]
Filename: "{app}\CodeInsight.exe"; Description: "启动CodeInsight"; Flags: nowait postinstall skipifsilent
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File \"{app}\install\post_install.ps1\" -Scope User"; Flags: runascurrentuser postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  // 检查Docker是否安装
  if not RegKeyExists(HKEY_LOCAL_MACHINE, 'SOFTWARE\Docker Inc.') then
  begin
    MsgBox('请先安装Docker Desktop后再运行CodeInsight安装程序。', mbError, MB_OK);
    Result := False;
  end 
  else
    Result := True;
end;

// 检查Docker Desktop是否正在运行
function InitializeUninstall(): Boolean;
var
  DockerProcess: String;
begin
  Result := True;
  
  // 检查Docker Desktop进程是否运行
  DockerProcess := 'docker-desktop.exe';
  if not IsProcessRunning(DockerProcess) then
  begin
    if MsgBox('Docker Desktop未运行。是否现在启动？', mbConfirmation, MB_YESNO) = IDYES then
    begin
      // 尝试启动Docker Desktop
      if not ShellExec('', 'docker-desktop.exe', '', '', SW_SHOWNORMAL, ewNoWait, Result) then
      begin
        MsgBox('无法启动Docker Desktop，请手动启动后再运行应用。', mbWarning, MB_OK);
      end;
    end;
  end;
end;

// 辅助函数：检查进程是否运行
function IsProcessRunning(FileName: String): Boolean;
var
  FileNameOnly: String;
  ProcessList: String;
begin
  FileNameOnly := ExtractFileName(FileName);
  ProcessList := GetProcessList;
  Result := Pos(FileNameOnly, ProcessList) > 0;
end;

// 辅助函数：获取进程列表
function GetProcessList: String;
var
  ProcessInfo: TProcessInformation;
  ProcessEntry: TProcessEntry32;
  Snapshot: THandle;
begin
  Result := '';
  Snapshot := CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if Snapshot <> INVALID_HANDLE_VALUE then
  begin
    ProcessEntry.dwSize := SizeOf(ProcessEntry);
    if Process32First(Snapshot, ProcessEntry) then
    begin
      repeat
        Result := Result + ProcessEntry.szExeFile + #13#10;
      until not Process32Next(Snapshot, ProcessEntry);
    end;
    CloseHandle(Snapshot);
  end;
end;

// 安装完成后显示信息
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssDone then
  begin
    MsgBox('CodeInsight AI代码分析平台安装完成！' + #13#10 + #13#10 + '程序已自动启动，您可以通过桌面图标或开始菜单访问应用。' + #13#10 + #13#10 + '如需停止服务，请使用系统托盘菜单或停止脚本。', 
           mbInformation, MB_OK);
  end;
end;

// 自定义页面：显示安装前信息
procedure InitializeWizard;
begin
  with WizardForm do
  begin
    Caption := 'CodeInsight AI代码分析平台 安装向导';
    WelcomeLabel1.Caption := '欢迎使用 CodeInsight AI代码分析平台 安装向导！';
    WelcomeLabel2.Caption := '此向导将引导您完成 CodeInsight AI代码分析平台的安装过程。';
    
    // 添加自定义信息
    CustomMessage := '安装前请确保：' + #13#10 + 
                    '• 已安装Docker Desktop' + #13#10 + 
                    '• 系统有足够的磁盘空间（至少2GB）' + #13#10 + 
                    '• 系统内存至少4GB' + #13#10 + #13#10 +
                    '安装完成后将自动启动应用。';
  end;
end;