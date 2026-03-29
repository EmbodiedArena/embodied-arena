# Simulation environments — setup

This document covers optional dependencies for embodied evaluation environments (manipulation, navigation, etc.).

---

## Gym

```bash
pip install gym==0.26.2
```

---

## Manipulation

### Loho-Ravens

```bash
git clone https://github.com/Shengqiang-Zhang/lohoravens.git
pip install -e . --no-deps  # cliport-0.1.0
export CLIPORT_ROOT=$(pwd)
```

**1. `omegaconf.errors.UnsupportedInterpolationType: Unsupported interpolation type env`**

Update `root_dir` in `cliport/cfg/config.yaml`:

```yaml
# root_dir: ${env.env:CLIPORT_ROOT}
root_dir: ${oc.env:CLIPORT_ROOT}
```

**2. `ImportError: The FFMPEG plugin is not installed`**

```bash
pip install 'imageio[ffmpeg]'
```

---

## Navigation

### AI2-THOR

```bash
pip install ai2thor==5.0.0
```

**1. `RuntimeError: vulkaninfo failed to run`**

Install Vulkan tools (example on Ubuntu):

```bash
sudo apt install vulkan-tools
```

**2. Headless / device tweaks**

See [allenai/ai2thor#1144](https://github.com/allenai/ai2thor/issues/1144) for community workarounds. One pattern is to customize how the Unity player is launched, e.g.:

```python
def unity_command(self, width, height, headless):
    fullscreen = 1 if self.fullscreen else 0

    command = self._build.executable_path

    if headless:
        command += " -batchmode -nographics"
    else:
        command += (
            " -screen-fullscreen %s -screen-quality %s -screen-width %s -screen-height %s"
            % (fullscreen, QUALITY_SETTINGS[self.quality], width, height)
        )

    if self.gpu_device is not None:
        # This parameter only applies to the CloudRendering platform.
        # Vulkan maps the passed-in value to device-index - 1 relative to
        # nvidia-smi device IDs.
        device_index = self.gpu_device if self.gpu_device < 1 else self.gpu_device + 1
        command += " -force-device-index %d" % device_index

    return shlex.split(command)
```

*(The snippet above is illustrative; patch the corresponding method in your installed `ai2thor` version if needed.)*
