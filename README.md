# VigilanceMonitor

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)

CameraPulse is a security monitoring system that leverages computer vision to ensure vigilance by detecting the presence of a person, the use of a phone, and camera tampering. It generates alerts based on defined conditions and stores them in a local database. Another script processes these alerts, uploads images to an FTP server, and sends data to an API.

## 📜 Features

- **Person Detection**: Alerts if no person is detected in the camera for `x` seconds.
- **Phone Detection**: Alerts if a phone is detected for more than `y` seconds.
- **Camera Tampering Detection**: Alerts if the camera is tampered with.
- **Local Database Storage**: Stores all alerts in a local database.
- **FTP Upload**: Uploads images stored locally to an FTP server.
- **API Integration**: Sends the data from the database to an API.

## 🛠️ Installation

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Sparsh57/VigilanceMonitor.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd VigilanceMonitor
    ```
3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1. **Running the Monitoring Script**:
    ```bash
    python Person_cell_optimized_2.py
    ```
    This script starts the monitoring process and generates alerts based on the defined conditions.

2. **Uploading to FTP and Sending Data to API**:
    ```bash
    python database_update.py
    ```
    This script reads the local database, uploads images to the FTP server, and sends the data to the API.

### Configuration

- Ensure the configuration settings for the camera, FTP server, and API are correctly set in the respective configuration files.

## 🤝 Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the repository**.
2. **Create a new branch**:
    ```bash
    git checkout -b feature-branch
    ```
3. **Commit your changes**:
    ```bash
    git commit -m 'Add new feature'
    ```
4. **Push to the branch**:
    ```bash
    git push origin feature-branch
    ```
5. **Create a new Pull Request**.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions or issues, please open an issue on GitHub or contact [Sparsh57](https://github.com/Sparsh57).

---

<p align="center">
    <img src="https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg" alt="Markdown Badge">
</p>
