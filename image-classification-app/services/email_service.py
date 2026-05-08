import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class EmailService:
    """Email alert service for fire/smoke detections"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.get('enabled', False)
    
    def send_alert(self, results, alert_count):
        """Send email alert for detections"""
        if not self.enabled:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']
            msg['Subject'] = f"🚨 Forest Fire Alert: {alert_count} Detections!"
            
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 20px; border-radius: 10px;">
                    <h1>🔥 Forest Fire Detection Alert</h1>
                    <p><strong>Time:</strong> {datetime.now():%Y-%m-%d %H:%M:%S}</p>
                </div>
                
                <div style="padding: 20px;">
                    <h2>Detection Summary</h2>
                    <p><strong>Total Alerts:</strong> {alert_count}</p>
                    <p style="color: red; font-size: 18px;">
                        <strong>⚠️ IMMEDIATE ATTENTION REQUIRED!</strong>
                    </p>
                    
                    <h3>Recent Detections:</h3>
                    <ul>
            """
            
            for r in results[:5]:
                body += f"""
                        <li>
                            <strong>{r['predicted_class'].upper()}</strong>: 
                            {r['confidence']:.1%} - {r['file']}
                        </li>
                """
            
            body += """
                    </ul>
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
                    <p>This is an automated alert from the Forest Fire Detection System.</p>
                    <p>Please take appropriate action immediately.</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False