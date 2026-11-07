"""Build optional buffer-only kernels; ordinary NumPy paths remain available."""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        from setuptools import Distribution, Extension

        root = Path(self.root)
        flags = ["/fp:strict"] if os.name == "nt" else ["-O3", "-ffp-contract=off"]
        extension = Extension("phenoradar._aa_kernels",
                              [str(root / "src/phenoradar/_aa_kernels.c")],
                              extra_compile_args=flags, optional=True)
        with TemporaryDirectory(prefix="phenoradar_build_") as temporary:
            command = Distribution({"ext_modules": [extension]}).get_command_obj("build_ext")
            command.build_lib = str(Path(temporary) / "lib")
            command.build_temp = str(Path(temporary) / "objects")
            command.ensure_finalized()
            command.run()
            path = Path(command.get_ext_fullpath(extension.name))
            target = root / "src/phenoradar" / path.name
            if path.exists():
                shutil.copy2(path, target)
            else:
                target.unlink(missing_ok=True)
        if target.exists():
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
            build_data["artifacts"].append(str(target.relative_to(root)))
