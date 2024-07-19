import os
import time
import datetime
import requests
from ftplib import FTP
import sqlite3

# Connect to the SQLite database

delete_query = "DELETE FROM Maindb WHERE filePath = ?"
# FTP server details
host = ""#ftp user name
username = ""#ftp username
password = ""#ftp password
ftp = FTP()

# Insert query for the database
insert_query = """
INSERT INTO Maindb ("companyCode", "exhibitionCode", "bootcode", "alertType", "filePath", "mimeType", "createdAt", "status") VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


# Function to upload file to FTP
def upload_to_ftp(row):
    global conn
    local_file = row[6]
    remote_dir = '/Maindb/alerts/' + str(row[8].split()[0])  # Remote directory based on createdAt date
    print(os.path.exists(local_file))
    if os.path.exists(local_file):
        try:
            # Connect to FTP server
            ftp.connect(host=host, port=2021)
            ftp.login(user=username, passwd=password)
            ftp.set_pasv(True)  # Enable passive mode
            # Change to remote directory (create if it doesn't exist)
            directories = remote_dir.strip('/').split('/')
            current_path = '/'
            for directory in directories:
                current_path = os.path.join(current_path, directory)
                try:
                    ftp.mkd(current_path)
                except Exception as e:
                    pass
                ftp.cwd(current_path)

            with open(local_file, 'rb') as file:
                ftp.storbinary(f"STOR {os.path.basename(local_file)}", file)

            # Close FTP connection
            ftp.quit()

            # Delete the local file after successful upload
            if os.path.exists(local_file):
                os.remove(local_file)

            # Return the FTP location for further use
            ftp_location = f"ftp://{host}/{remote_dir}/{os.path.basename(local_file)}"
            return ftp_location

        except Exception as e:
            return None
    else:
        conn.execute(delete_query, (local_file,))
        conn.commit()


# Function to send data to API
def send_data_to_api(row, ftp_location):
    try:
        api_url = ""#api url
        created_at = datetime.datetime.strptime(row[8], '%Y-%m-%d %H:%M:%S.%f').isoformat()
        api_data = {
            "company_code": row[2],
            "exhibition_code": row[3],
            "booth_code": row[4],
            "alert_type": row[5],
            "dateandtime": created_at,
            "filepath": ftp_location,
            "mime_type": row[7],
            "alert_status": "pending",
        }
        response = requests.post(api_url, json=api_data)
        if response.status_code == 200:
            print("Data sent to API successfully.")
            return True
        else:
            print(f"Failed to send data to API. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(e)
        return False


# Function to check and update the database
def check_and_update_database():
    global conn
    try:
        conn = sqlite3.connect('maindatabase.db')
        print(conn)
    except Exception as e:
        print("Error. Database doesn't exist", e)
        raise
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT rowid, * FROM Maindb WHERE status = 'N'")
        rows = cursor.fetchall()
        for row in rows:
            print(f"Processing row ID: {row[0]}")
            ftp_location = upload_to_ftp(row)
            if ftp_location:
                success = send_data_to_api(row, ftp_location)
                if success:
                    conn.execute("UPDATE Maindb SET filePath = ?, status = 'Y' WHERE rowid = ?",
                                 (ftp_location, row[0]))
                    conn.commit()
    except Exception as e:
        print(e)
        conn.rollback()


# Main loop to continuously check and update the database
while True:
    check_and_update_database()
    time.sleep(10)
