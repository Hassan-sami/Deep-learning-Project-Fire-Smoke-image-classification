import json
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict

class ReportService:
    """Generate reports from detection results"""
    
    @staticmethod
    def _normalize_result(result):
        """Normalize result keys to handle different data formats"""
        return {
            'file': result.get('file') or result.get('image', 'unknown'),
            'predicted_class': result.get('predicted_class') or result.get('prediction', 'unknown'),
            'confidence': result.get('confidence', 0),
            'inference_time': result.get('inference_time', 0),
            'above_threshold': result.get('above_threshold', True),
            'timestamp': result.get('timestamp', '')
        }
    
    @staticmethod
    def generate_html_report(results, output_path, title="Fire Detection Report"):
        """Generate HTML report"""
        # Normalize all results
        normalized_results = [ReportService._normalize_result(r) for r in results]
        confidences = [r['confidence'] for r in normalized_results]
        
        # Replace Unicode characters with HTML entities to avoid encoding issues
        fire_count = sum(1 for r in normalized_results if r['predicted_class'].lower() == 'fire')
        smoke_count = sum(1 for r in normalized_results if r['predicted_class'].lower() == 'smoke')
        high_conf_count = sum(1 for c in confidences if c > 0.8)
        avg_time = np.mean([r.get('inference_time', 0) for r in normalized_results]) if normalized_results else 0
        
        # Alert detections
        alerts = [r for r in normalized_results if r['predicted_class'].lower() in ['fire', 'smoke']]
        alert_count = len(alerts)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {datetime.now():%Y-%m-%d %H:%M}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f6fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header p {{ margin: 5px 0; opacity: 0.9; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        .stat-box h3 {{ margin: 0; color: #7f8c8d; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
        .stat-box .value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .fire {{ color: #e74c3c; }}
        .smoke {{ color: #f39c12; }}
        .safe {{ color: #27ae60; }}
        .alert-box {{ background: #ffeaa7; border-left: 5px solid #fdcb6e; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .alert-box strong {{ color: #856404; }}
        table {{ width: 100%; background: white; border-collapse: collapse; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 20px; }}
        th {{ background: #2c3e50; color: white; padding: 15px; text-align: left; font-size: 14px; text-transform: uppercase; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .confidence-bar {{ height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }}
        .confidence-fill {{ height: 100%; border-radius: 10px; transition: width 0.3s; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; text-transform: uppercase; }}
        .badge-fire {{ background: #ffe0e0; color: #e74c3c; }}
        .badge-smoke {{ background: #fff3cd; color: #f39c12; }}
        .badge-safe {{ background: #d4edda; color: #27ae60; }}
        .badge-uncertain {{ background: #e2e3e5; color: #6c757d; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #7f8c8d; font-size: 14px; }}
        .section-title {{ color: #2c3e50; margin: 30px 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
            <p>Total Images Processed: {len(normalized_results)}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Total Images</h3>
                <div class="value">{len(normalized_results)}</div>
            </div>
            <div class="stat-box">
                <h3>Fire Detections</h3>
                <div class="value fire">{fire_count}</div>
            </div>
            <div class="stat-box">
                <h3>Smoke Detections</h3>
                <div class="value smoke">{smoke_count}</div>
            </div>
            <div class="stat-box">
                <h3>Avg Confidence</h3>
                <div class="value">{np.mean(confidences):.1%}</div>
            </div>
            <div class="stat-box">
                <h3>High Confidence</h3>
                <div class="value safe">{high_conf_count}</div>
            </div>
            <div class="stat-box">
                <h3>Avg Time</h3>
                <div class="value">{avg_time:.3f}s</div>
            </div>
        </div>
"""
        
        # Alert box for fire/smoke detections
        if alert_count > 0:
            urgency = "Immediate attention required!" if alert_count > 5 else "Please review the detections."
            html += f"""
        <div class="alert-box">
            <strong>WARNING: {alert_count} fire/smoke detections found!</strong><br>
            {urgency}
        </div>
"""
        
        # Results table
        html += """
        <h2 class="section-title">Detection Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Image</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Confidence Bar</th>
                    <th>Time (s)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for i, r in enumerate(normalized_results, 1):
            pred_class = r['predicted_class']
            confidence = r['confidence']
            
            # Determine badge class
            if pred_class.lower() == 'fire':
                badge_class = 'badge-fire'
            elif pred_class.lower() == 'smoke':
                badge_class = 'badge-smoke'
            elif pred_class.lower() == 'non_fire':
                badge_class = 'badge-safe'
            else:
                badge_class = 'badge-uncertain'
            
            # Confidence bar color
            if confidence > 0.8:
                bar_color = '#27ae60'
            elif confidence > 0.6:
                bar_color = '#f39c12'
            else:
                bar_color = '#e74c3c'
            
            # Status indicator
            status = 'PASS' if r.get('above_threshold', True) else 'LOW'
            status_color = '#27ae60' if status == 'PASS' else '#e74c3c'
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{r['file']}</strong></td>
                    <td><span class="badge {badge_class}">{pred_class.upper()}</span></td>
                    <td>{confidence:.2%}</td>
                    <td>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence*100}%; background: {bar_color};"></div>
                        </div>
                    </td>
                    <td>{r.get('inference_time', 0):.3f}</td>
                    <td><span style="color: {status_color}; font-weight: bold;">{status}</span></td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Forest Fire Detection System - Automated Report</p>
            <p>Generated by AI-Powered Detection Engine</p>
        </div>
    </div>
</body>
</html>"""
        
        # Write file with UTF-8 encoding explicitly
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(output_path)
    
    @staticmethod
    def generate_json_report(results, output_path):
        """Generate JSON report"""
        # Normalize results
        normalized_results = [ReportService._normalize_result(r) for r in results]
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_results': len(normalized_results),
            'results': normalized_results,
            'summary': {
                'fire_count': sum(1 for r in normalized_results if r['predicted_class'].lower() == 'fire'),
                'smoke_count': sum(1 for r in normalized_results if r['predicted_class'].lower() == 'smoke'),
                'non_fire_count': sum(1 for r in normalized_results if r['predicted_class'].lower() == 'non_fire'),
                'uncertain_count': sum(1 for r in normalized_results if r['predicted_class'].lower() == 'uncertain'),
                'average_confidence': np.mean([r['confidence'] for r in normalized_results]) if normalized_results else 0,
                'average_time': np.mean([r.get('inference_time', 0) for r in normalized_results]) if normalized_results else 0,
                'high_confidence_count': sum(1 for r in normalized_results if r['confidence'] > 0.8)
            }
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(output_path)