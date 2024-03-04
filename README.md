Windows Executable for [https://github.com/karaokenerds/python-audio-separator/](https://github.com/karaokenerds/python-audio-separator/) using PyInstaller withg Cuda.

INSTALL WITH CONDA:

`conda env create -f environment.yml`

COMPILE TO WINDOWS EXECUTABLE:

`pyinstaller .\main.py --clean --add-binary "C:\Users\User\miniconda3\envs\audio_separator_cuda_windows_exe\Lib\site-packages\onnxruntime\capi\*;onnxruntime/capi/" --add-data ".\simple_model.onnx:." --noconfirm --collect-all audio_separator`

USAGE

`.\dist\main\main.exe <python-audio-separator params>`
