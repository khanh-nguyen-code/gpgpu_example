# OPENCL_EXAMPLE

## INSTALLATION

- PoCL - Portable Computing Language - open source implementation on CPU

```shell
# fedora dnf
sudo dnf install pocl
# arch linux pacman
sudo pacman -S pocl
# macos homebrew
brew install pocl
```

- Rocm - AMD implementation

```shell
# fedora dnf
sudo dnf install rocmclinfo rocm-*
# arch linux paru
paru -S rocm-opencl-runtime # https://wiki.archlinux.org/title/GPGPU#AMD/ATI
```

- Apple - Xcode Command Line Tools
```shell
xcode-select --install
```

OpenCL implementation will be located at `/System/Library/Frameworks/OpenCL.framework`
