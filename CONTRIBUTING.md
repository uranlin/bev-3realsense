# 開發指南

使用 Python 3.12，安裝 `requirements-dev.txt`。核心測試不需要相機、GPU 或模型權重。

```sh
python -m pytest -q
python -m ruff check backend tests tools server.py
python -m ruff format --check backend tests tools server.py
```

幾何運算的變更應附上能重現邊界條件的測試；API 變更應驗證成功與失敗回應。硬體測試須記錄相機型號、串流設定、校正資料及結果。

提交以單一目的為單位，訊息描述行為或結構的改變。不要提交虛擬環境、模型權重、錄製資料、帳號憑證與個人路徑。技術限制寫在對應文件，不把本機操作過程放入 README。

前端使用原生 HTML、CSS 與 JavaScript，無需建置。格式化可使用 `npx prettier@3.6.2 --write frontend`。
