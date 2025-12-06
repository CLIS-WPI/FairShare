# Dev Container Configuration

این devcontainer برای VS Code استفاده می‌شه.

## 🚀 استفاده

1. VS Code رو باز کنید
2. `F1` → "Dev Containers: Reopen in Container"
3. VS Code container رو build و start می‌کنه

## 📋 فایل‌ها

- **`devcontainer.json`** - VS Code devcontainer config
- **`post-create.sh`** - Script که بعد از create container اجرا می‌شه

## 🔧 تنظیمات

- **Base Image:** `docker/Dockerfile.final`
- **GPU Support:** `--gpus=all`
- **Workspace:** `/workspace`
- **Python:** 3.12
- **Extensions:** Python, Pylance, YAML, Docker, GitLens, Jupyter

---

**برای Docker manual:** `docker/README.md`

