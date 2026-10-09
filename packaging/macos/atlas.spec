# PyInstaller spec for Atlas.app (macOS). Build with build_macos.sh.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent.parent

datas = [
    (str(ROOT / "config" / "settings.yaml"), "config"),
    (str(ROOT / "yolov8s.pt"), "."),
]
datas += collect_data_files("ultralytics")

hiddenimports = collect_submodules("ultralytics")

a = Analysis(
    [str(ROOT / "src" / "app_entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(Path(SPECPATH) / "hooks")],
    excludes=["pytest", "tkinter", "IPython", "notebook", "scripts", "tests", "polars", "polars_runtime_32"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Atlas",
    console=False,
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name="Atlas", upx=False)

app = BUNDLE(
    coll,
    name="Atlas.app",
    bundle_identifier="com.atlas.assistivevision",
    info_plist={
        "CFBundleName": "Atlas",
        "CFBundleDisplayName": "Atlas",
        "CFBundleShortVersionString": "1.0.0",
        "NSCameraUsageDescription": "Atlas uses the camera to analyze nearby objects and movement.",
        "NSHighResolutionCapable": True,
    },
)
