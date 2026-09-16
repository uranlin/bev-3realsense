# 架構

## 模組責任

| 模組 | 責任 |
|---|---|
| api/app.py | 建立 FastAPI、綁定 runtime、掛載靜態頁 |
| api/schemas.py | 驗證請求資料與數值範圍 |
| api/routes.py | HTTP 輸入、服務呼叫、回應與錯誤 |
| services/runtime.py | 管理相機、模型、背景工作及最新場景 |
| models/ | 將外部 SDK／推論模型轉為影像與深度陣列 |
| processing/ | 投影、格網運算、路徑搜尋與渲染 |
| utils/ | 矩陣建構、內參縮放與圖像編碼 |

讀程式時可從 `server.py` → `api/app.py` → `api/routes.py` 開始。幾何運算可直接閱讀 `processing/bev.py` 與 `processing/planner.py`，不需要先理解 HTTP 層。

## 資料流

```text
RGB-D → 深度有效性檢查 → BEV 投影 → 多相機融合
                                      ↓
                              障礙／自由／未知格網
                                      ↓
                              公尺尺度膨脹 → A*
                                      ↓
                              世界座標路徑與渲染圖
```

API 與 runtime 使用 NumPy 格網；HTTP 回傳 JSON 與 base64 PNG。合成場景直接產生 BEV 觀測，用於離線驗證。

## 座標約定

- 相機：X 向右、Y 向下、Z 向前，單位為公尺。
- 預設格網：500×500，X ∈ [-4,4)、Z ∈ [0.3,5)，遠方位於圖上方。
- 世界座標路徑使用格子中心；超出地圖的目標回傳 422。
- A* 只搜尋已觀測的可通行格，不穿越障礙角落，也不自行挪動終點。

## 狀態與資源

每個應用程式工廠建立一個 `NavigationRuntime`，存於 `app.state.runtime`。停止服務時關閉相機與背景工作。變更設定會清除舊地圖。

格網設定仍由 `backend/config.py` 共用。目前設計範圍為單程序、單人操作；不應將多個不同設定的應用實例放在同一程序。多使用者狀態隔離與時間戳管理尚未實作。
