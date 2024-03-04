
# ✅ WORKING TORCH + ONNX WITH CUDA:
# https://stackoverflow.com/questions/71269142/onnxruntime-gpu-failing-to-find-onnxruntime-providers-shared-dll-when-run-from-a
# pyinstaller .\main.py --clean --add-binary "C:\Users\User\miniconda3\envs\test-pyinstaller-cuda-torch\Lib\site-packages\onnxruntime\capi\*;onnxruntime/capi/" --noconfirm --add-data "./ffmpeg/*;./ffmpeg/"
# MORE COMMANDS: https://pyinstaller.org/en/v4.1/usage.html#what-to-generate

# CHECK WHICH DLLS ARE BEING USED BY APP WITH listdlls cmd https://download.sysinternals.com/files/ListDlls.zip
# PEGA PID do executable e da run no terminal. da para ver q usa as dll do driver da placa.


import numpy as np
import torch


# Check if CUDA is available
if torch.cuda.is_available():
    # Set device to the first CUDA device
    cuda_device = torch.device("cuda:0")
    print("✅ CUDA device available, using GPU:", torch.cuda.get_device_name(0))
    
    # Create tensors
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)

    # Move tensors to GPU
    x_gpu = x.to(cuda_device)
    y_gpu = y.to(cuda_device)

    # Perform matrix multiplication on GPU
    z_gpu = torch.matmul(x_gpu, y_gpu)

    # Move the result back to CPU
    z_cpu = z_gpu.to("cpu")

    # Print the result
    print("✅ Result of matrix multiplication:", z_cpu)
else:
    print("❌ CUDA is not available. Please check your setup.")


# if onxxruntime device is GPU then log and test
import onnxruntime

    # Check if ONNX Runtime can use CUDA
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    
    print("✅ ONNX Runtime: CUDA is available.")

    # TESTAR EXECUÇÃO
    session = onnxruntime.InferenceSession("simple_model.onnx", providers=['CUDAExecutionProvider'])
    print("✅ Running on CUDA")

    # Create dummy input data as a numpy array
    dummy_input = np.random.randn(1, 10).astype(np.float32)

    # Run the model (forward pass)
    inputs = {session.get_inputs()[0].name: dummy_input}
    outputs = session.run(None, inputs)

    print("✅ Output from OnnxRuntime GPU: ", outputs[0])

    # Example to showcase ONNX Runtime CUDA capabilities would typically involve loading an ONNX model
    # and performing inference. Here, we simply log the availability.
    print("✅ ONNX Runtime can utilize the GPU for models. Device: ", onnxruntime.get_device())
else:
    print("❌ ONNX Runtime: CUDA is not available. Please check your setup.")


# from audio_separator.separator import Separator
# import logging

# # Initialize the Separator class (with optional configuration properties below)
# separator = Separator(log_level=logging.INFO)

# # Load a machine learning model (if unspecified, defaults to 'UVR-MDX-NET-Inst_HQ_3.onnx')
# separator.load_model()

# # Perform the separation on specific audio files without reloading the model
# primary_stem_output_path, secondary_stem_output_path = separator.separate('feijaopuro.wav')

# print(f'Primary stem saved at {primary_stem_output_path}')
# print(f'Secondary stem saved at {secondary_stem_output_path}')
    
import audio_separator.utils.cli
audio_separator.utils.cli.main()
