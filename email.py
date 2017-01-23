import sys
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

fromaddr = "my.alerts.jesus.vergara@gmail.com"
toaddr = "eejvt@leeds.ac.uk"
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Script finished"
 
body = "Your script \n %s  \n has finished "%sys.argv[1]
msg.attach(MIMEText(body, 'plain'))
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, "palomaSS")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()
