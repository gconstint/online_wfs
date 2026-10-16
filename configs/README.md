# 实验配置

| 配置 | 数据 | 用途 | Y / X 像素尺寸 [µm] | crop [y0,y1,x0,x1] |
|---|---|---|---|---|
| [run11.json](run11.json) | `run11/img_010.tif` | slit 参考图，M2 像素标定 | 0.48 / 0.48（初始值） | [1455,4692,2268,2965] |
| [run13.json](run13.json) | `run13/img_420.tif` | slit 全开，波前重建与焦距计算 | 0.483928694 / 0.483928694 | [864,4644,2268,2616] |

两组配置均为 10000 eV、slit/参考源到光栅 154.2 m、光栅到探测器 0.681 m、
标称光栅周期 18.38 µm、光栅角度 45°。
JSON 中 `pixel_size` 的顺序为 `[py, px]`，单位为米；距离和周期也用米。
crop 的结束索引不包含在内，run11 的裁剪尺寸为 `(3237,697)`，run13 为 `(3780,348)`。
M2 支持每轴不足 8 个预期条纹周期的 ROI；当前实现最低支持 2 个周期，仍检查峰的可辨识性。
run13 的双轴像素值沿用用户指定值，与 run11 分轴标定结果分别保存。

run11 当前 crop 按 Y 像素尺寸目标 `0.48393031 µm` 选取，单峰轴向投影计算结果为
Y `0.483930313 µm`、X `0.483925800 µm`；Y 保留 8 位小数与目标一致。
目标值没有写入标定算法，几何参数和初始像素值也未调整。
这是按目标选区后的结果，不能将数值吻合本身作为标定精度的独立验证。

在仓库根目录运行：

```bash
python cal_pixel_size.py --config configs/run11.json
python cal_pixel_size.py --config configs/run11.json --plot
python main.py --config configs/run13.json
```

标定脚本默认只打印载频比较表和像素尺寸变化，不改写配置或生成结果文件。
加 `--plot` 显示 ROI 图像、FFT 峰位及 X/Y 校正像素，不自动保存图像。
显式命令行参数覆盖配置值，例如 `--pixel-size-um PY PX`、`--image ...`、
`--crop-indexes Y0 Y1 X0 X1`；`--no-crop` 关闭配置中的裁剪。
`main.py` 按现有流程保存重建结果和图像，可通过 `--output-dir` 指定结果目录。
examples 中的示例仍通过 parser 直接接收参数。

图像路径使用当前实验数据的绝对路径；若改成相对路径，则以配置文件目录为基准。
可选的 dark/flat 路径与 crop 沿用 `tools.io.load_inputs` 的读取规则。
dark/flat 只在显式提供路径时校正；可在配置中设置 `dark_image_path` / `flat_image_path`，
也可使用对应命令行参数。
