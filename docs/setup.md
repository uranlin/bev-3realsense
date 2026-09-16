# 安裝與執行

## 核心環境

Python 3.12，支援 Windows、Linux 與 macOS 的離線模式。

```sh
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements-core.txt
.venv\Scripts\python server.py
# Linux / macOS
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python server.py
```

網頁：http://127.0.0.1:8000 。API 文件：http://127.0.0.1:8000/docs 。

## 模式

| BEV_MODE | 安裝 | 用途 |
|---|---|---|
| `demo` | requirements-core.txt | 離線合成 BEV，不下載模型或存取相機 |
| `realsense` | 核心 + opencv-python、pyrealsense2 | 單台或多台相機單次擷取 |
| `full` | requirements.txt + 適合硬體的 PyTorch | 相機、單目深度及 YOLO 後端 |

PowerShell：

```powershell
.\.venv\Scripts\python -m pip install opencv-python pyrealsense2
./start.ps1 -Mode realsense
```

Linux/macOS：`BEV_MODE=realsense .venv/bin/python server.py`。

`BEV_DEVICE` 預設 `cpu`；安裝可用的 CUDA PyTorch 後可設為 `cuda`。完整模型模式尚未完成獨立環境與 GPU 驗證，首次使用會下載深度模型。YOLO 權重存放於 `weights/yolov8n.pt`；該目錄不納入版本控制。

目前網頁提供場景載入、相機單次擷取與規劃。圖片上傳及模型切換使用 API。更改相機設定後必須重新擷取。
