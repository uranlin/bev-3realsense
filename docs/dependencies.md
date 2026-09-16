# 第三方依賴

核心模式使用 FastAPI、Uvicorn、NumPy、SciPy 與 Pillow。硬體模式另使用 OpenCV 與 librealsense；完整模式包含 PyTorch、Transformers、Depth Anything V2 與 Ultralytics。

模型權重不隨 Git 原始碼分發。部署或散布前，請核對實際版本及模型來源的授權條款。

- [librealsense](https://github.com/realsenseai/librealsense)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [SciPy](https://github.com/scipy/scipy)

技術參考：[RealSense 投影與座標](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)、[SciPy 距離轉換](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)。
