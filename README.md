# 預測學習（Predictive Learning）

本儲存庫整理預測學習課程資料與程式作業。目前已完成 **Homework 1：Risk Minimization and Model Selection**，包含跨年 KNN 分類、迴歸模型選擇及五折交叉驗證。

## 專案架構

```text
.
├─ README.md                 # 專案總覽與執行方式
├─ .gitignore
├─ course/                   # 課程大綱與講義 PDF
└─ HW1/
   ├─ README.md             # 作業說明與結果速覽
   ├─ report.md             # 中文實驗報告
   ├─ Homework1_Report_EN.pdf  # 英文正式報告
   ├─ Homework1.pdf         # 作業題目
   ├─ main.py               # 四題實驗的共同執行入口
   ├─ hw1.py                # 舊 IDE 入口，轉呼叫 main.py
   ├─ requirements.txt      # 固定版本的 Python 相依套件
   ├─ data/                 # 2000、2004 選舉資料與迴歸樣本
   ├─ src/                  # 資料載入、模型、評估、繪圖與報告
   ├─ results/
   │  ├─ tables/           # CSV 數值與逐筆預測
   │  └─ figures/          # 實驗圖
   └─ tests/                # pytest 測試
```

[HW1 作業說明](HW1/README.md)介紹四題的重點；[中文報告](HW1/report.md)提供實驗方法、數值、圖表與限制；[英文報告](HW1/Homework1_Report_EN.pdf)是正式撰寫版本。

## 使用虛擬環境執行

建議使用 **Python 3.11**，在儲存庫根目錄建立虛擬環境。以下為 Windows PowerShell 指令；電腦需先安裝 Python 3.11 與 [uv](https://docs.astral.sh/uv/)：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .\.venv\Scripts\python.exe -r HW1\requirements.txt
.\.venv\Scripts\python.exe HW1\main.py
.\.venv\Scripts\python.exe -m pytest HW1\tests -q
```

若已經建立根目錄的 `.venv`，可略過前兩行。沒有 uv 時，可用系統的 Python 建立虛擬環境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r HW1\requirements.txt
.\.venv\Scripts\python.exe HW1\main.py
.\.venv\Scripts\python.exe -m pytest HW1\tests -q
```

在 macOS 或 Linux 上，虛擬環境的執行檔路徑改為 `.venv/bin/python`。執行 `HW1/main.py` 會重建 [結果表格](HW1/results/tables/)、[圖表](HW1/results/figures/)及 [中文報告](HW1/report.md)。固定的套件版本列在 [HW1/requirements.txt](HW1/requirements.txt)。

## 版本庫內容與注意事項

- [course/](course/) 存放課程參考 PDF；實驗程式與資料集中在 [HW1/](HW1/)。
- [HW1/data/](HW1/data/) 的原始選舉 CSV 與 [HW1/results/](HW1/results/) 的輸出目前保留在版本庫，方便直接閱讀及比對實驗結果。
- `.venv/`、`.pytest_cache/`、`__pycache__/` 等本機環境或測試快取由 [.gitignore](.gitignore) 排除，不需上傳 GitHub。pytest 自動產生的 `.pytest_cache/README.md` 不是本專案的說明文件。
- 執行環境若不同，鄰居等距時的選取及浮點數輸出可能受套件版本影響；請參照報告中的實驗條件。報告對應目前固定資料與設定，修改資料或演算法後應重新核對文字與數值。
