import marimo

__generated_with = "0.6.22"
app = marimo.App(width="medium")


@app.cell
def __():
    # Dependencies for starting llamafile and phoenix servers
    import subprocess
    import sys
    import os
    import atexit
    import signal

    # Typing and annotation dependencies
    from typing import List, Union
    from enum import Enum
    from dataclasses import dataclass, field
    return (
        Enum,
        List,
        Union,
        atexit,
        dataclass,
        field,
        os,
        signal,
        subprocess,
        sys,
    )


@app.cell
def __(dataclass, field, os, subprocess):
    class Server:
        """Start the server subprocess based on the command passed to it"""

        @dataclass
        class LLamafile:
            exact_model_name: str
            path_to_dir: str = "/Users/hazn/Desktop/code.nosync/llamafile"
            absolute_path: str = field(
                init=False
            )  # This field won't be a parameter in __init__

            def __post_init__(self):
                """Set the absolute path to the llamafile"""
                # check if path_to_dir is a valid path
                # if not, raise an error
                if not os.path.exists(self.path_to_dir):
                    raise FileNotFoundError(f"{self.path_to_dir} not found")

                if not self.exact_model_name.endswith(".llamafile"):
                    temp_name_with_extension = f"{self.exact_model_name}.llamafile"

                self.absolute_path = os.path.join(self.path_to_dir, temp_name_with_extension)

        @dataclass
        class Phoenix:
            pass

        type Command = LLamafile | Phoenix

        @staticmethod
        def start_server(cmd: Server.Command) -> subprocess.Popen:
            match cmd:
                case Server.LLamafile():
                    # ensure that the llamafile is executable
                    subprocess.Popen(f"chmod 755 {cmd.absolute_path}", shell=True, text=True)
                    return subprocess.Popen(
                        cmd.absolute_path,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )

                case Server.Phoenix():
                    return subprocess.Popen(
                        "mix phx.server",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
    return Server,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
