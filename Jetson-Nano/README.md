Create file to auto run the button manager on boot

``` bash
sudo vi /etc/systemd/system/facerec-manager.service
```

Add the following content to the file:

```ini
[Unit]
Description=Face Recognition Button Manager
After=network.target

[Service]
ExecStart=/usr/bin/python3 <link_to_button_manager.py>
WorkingDirectory=<link_to_project_directory>
StandardOutput=inherit
StandardError=inherit
Restart=always
User=vanwtoanf

[Install]
WantedBy=multi-user.target
```

``` bash
sudo systemctl daemon-reload
sudo systemctl enable facerec-manager.service
sudo systemctl start facerec-manager.service
``` 