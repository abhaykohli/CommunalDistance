import smtplib, ssl


class Mailer:
    def __init__(self):
        # Configure your credentials below
        self.email = ""
        self.password = ""
        # Specify target server SMTP port no.
        self.port = 465
        # Connect to GMail SMTP server
        self.server = smtplib.SMTP_SSL("smtp.gmail.com", self.port)

    def send(self, targetEmail):
        # Connects to gmail smtp server
        self.server = smtplib.SMTP_SSL("smtp.gmail.com", self.port)
        # Login using credentials
        self.server.login(self.email, self.password)

        # Message to be sent
        subject = "Alert!"
        body = f"Social distancing violations exceeded!"
        message = "Subject: {}\n\n{}".format(subject, body)

        # Send email
        self.server.sendmail(self.email, targetEmail, message)
        self.server.quit()
