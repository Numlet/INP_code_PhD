# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:04:30 2016

@author: eejvt
"""

#my.alerts.jesus.vergara@gmail.com 
import sys
import Jesuslib as jl
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
#server = smtplib.SMTP('smtp.gmail.com', 587)
#server.starttls()
#server.login("my.alerts.jesus.vergara@gmail.com ", "palomaSS")

jl.send_email()
#%%
#fromaddr = "my.alerts.jesus.vergara@gmail.com"
#toaddr = "eejvt@leeds.ac.uk"
#msg = MIMEMultipart()
#msg['From'] = fromaddr
#msg['To'] = toaddr
#msg['Subject'] = "Script finished"
# 
#body = "Your script \n %s  \n has finished "%sys.argv[0]
#msg.attach(MIMEText(body, 'plain'))
#server = smtplib.SMTP('smtp.gmail.com', 587)
#server.starttls()
#server.login(fromaddr, "palomaSS")
#text = msg.as_string()
#server.sendmail(fromaddr, toaddr, text)
#server.quit()
#%%
#msg = "test message"
#server.sendmail("my.alerts.jesus.vergara@gmail.com", "my.alerts.jesus.vergara@gmail.com", msg)
#server.quit()