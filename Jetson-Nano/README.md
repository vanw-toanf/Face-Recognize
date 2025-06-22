create sudo vi /etc/systemd/system/facerec-manager.service
'''
[Unit]
Description=Face Recognition Button Manager
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/vanwtoanf/project/face-recognize/button_manager.py
WorkingDirectory=/home/vanwtoanf/project/face-recognize
StandardOutput=inherit
StandardError=inherit
Restart=always
User=vanwtoanf

[Install]
WantedBy=multi-user.target
'''