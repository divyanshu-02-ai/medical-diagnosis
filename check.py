import torch

def main():
    print("PyTorch version:", torch.__version__)

    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if cuda_available:
        print("CUDA version (PyTorch):", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))

        # Tensor test on GPU
        x = torch.rand(3, 3).cuda()
        print("Tensor device:", x.device)
    else:
        print("Running on CPU")

if __name__ == "__main__":
    main()