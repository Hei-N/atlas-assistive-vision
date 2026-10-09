# macOS app build

Builds a self-contained `Atlas.app` with PyInstaller (bundled Python, dependencies,
`config/settings.yaml` and `yolov8s.pt`). For developers; end users only open the app.

## Build

```bash
source venv/bin/activate
pip install pyinstaller          # once
./packaging/macos/build_macos.sh
```

Output: `dist/Atlas.app` (about 600 MB). `yolov8s.pt` must be in the repository root.

## Test

```bash
open dist/Atlas.app
```

macOS asks for camera access on first launch. Logs are written to
`~/Library/Logs/Atlas/atlas.log`. A startup failure (for example no camera) shows
a dialog.

To run the bundle on a video file instead of the camera:

```bash
dist/Atlas.app/Contents/MacOS/Atlas --source path/to/video.mp4
```

## Notes

- The app is ad-hoc signed only: not signed with a developer identity or notarized.
  Gatekeeper will warn on other Macs (right-click, Open).
- Built for the architecture of the machine that builds it (arm64 here).
- Runtime-generated files go to `~/Library/Logs/Atlas/` and
  `~/Library/Application Support/Atlas/`, never into the app bundle.
