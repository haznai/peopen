import marimo

__generated_with = "0.4.6"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import torch
    return mo, torch


@app.cell
def __(torch):
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

    else:
        mps_device = torch.device("mps")

        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        # Or
        x = torch.ones(5, device="mps")

        # Any operation happens on the GPU
        y = x * 2

        # Move your model to mps just like any other device
        model = torch.nn.Linear(5, 2)
        model.to(mps_device)

        # Now every call runs on the GPU
        pred = model(x)
        print(pred)
    return model, mps_device, pred, x, y


if __name__ == "__main__":
    app.run()
