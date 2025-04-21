
# Hailo AI Lab

A lightweight web application for running AI inference on Raspberry Pi using the Hailo AI processor.

## 🧰 Prerequisites

- Raspberry Pi with Hailo driver and `hailo-all` installed
- Python 3.10
- Git
- Internet connection to download resources

## 📁 Setup Instructions

1. **Create and enter project directory**

```bash
mkdir Hailo-AI-Lab
cd Hailo-AI-Lab
```

2. **Create a virtual environment**

```bash
python -m venv --system-site-packages env
source env/bin/activate
```

3. **Clone the project repository**

```bash
git clone https://github.com/KasunThushara/AI-LAB.git
cd AI-LAB
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Download required resources**

```bash
chmod +x download_resources.sh
./download_resources.sh
```

## 🚀 Running the App

1. **Open a new terminal window**

```bash
cd Hailo-AI-Lab
source env/bin/activate
cd AI-LAB
python3 app.py
```

2. **Access the web app**

Open a browser and navigate to:

```
http://<pi-ip-address>:5000
```

Replace `<pi-ip-address>` with the IP address of your Raspberry Pi.

---

## 🧠 Features

- Object detection using Hailo AI chip
- Web interface for real-time camera streaming
- Light-weight and optimized for embedded devices

## 📂 Project Structure

```
Hailo-AI-Lab/
├── env/                      # Python virtual environment
└── AI-LAB/
    ├── app.py                # Main web application
    ├── requirements.txt      # Python dependencies
    ├── download_resources.sh # Resource downloader
    └── ...                   # Other model and utility files
```

