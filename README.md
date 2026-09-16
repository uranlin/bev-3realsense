# BEV Navigation

以 RealSense RGB-D 影像建立鳥瞰格網，融合多相機觀測，提供自由空間分類與 A* 路徑規劃。

提供兩種主要輸入：離線合成場景，以及一至三台 RealSense 相機。網頁介面可查看融合結果、設定目標與機器人半徑。

## 啟動

需要 Python 3.12。於專案根目錄執行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-core.txt
.\.venv\Scripts\python server.py
```

開啟 [localhost:8000](http://127.0.0.1:8000)。預設載入離線場景，按「規劃路徑」即可查看結果。再次啟動可使用 `./start.ps1`。

相機安裝、完整模型模式與環境變數見 [安裝說明](docs/setup.md)。

## 目錄

```text
backend/
  api/            應用程式工廠、HTTP 路由、請求格式
  services/       相機生命週期、推論與目前場景狀態
  models/         RealSense、單目深度與物件偵測介面
  processing/     BEV 投影、融合、分類、規劃與渲染
  utils/          相機矩陣與影像編碼
  config.py       格網、深度與分類預設值
  demo.py         合成 BEV 觀測
frontend/         HTML、CSS、JavaScript
tools/            相機診斷工具
tests/            幾何與 HTTP 測試
docs/             架構、安裝及硬體限制
server.py         本機啟動入口
```

[架構與資料流](docs/architecture.md) · [硬體驗證](docs/hardware.md) · [開發指南](CONTRIBUTING.md)

## 測試

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check backend tests tools server.py
```

## 使用範圍

這是一個感知與路徑規劃原型，沒有馬達控制輸出。未知格網不可通行；缺乏足夠觀測時，規劃器會回報無路徑。

離線場景用於驗證融合與規劃，不模擬相機深度誤差。目前尚未完成三相機外參校正、時間同步及實機導航驗收。

## 授權

本專案採用 [GNU Affero General Public License v3.0](LICENSE)（AGPL-3.0-only）。第三方套件及模型權重遵循各自條款，見 [第三方依賴](docs/dependencies.md)。
