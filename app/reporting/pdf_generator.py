import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

class PDFGenerator:
    @staticmethod
    def generate_report(file_name, profile, stats, ml_result=None):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        elements.append(Paragraph(f"NEXUS AI - Automated Data Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # 1. Dataset Overview
        elements.append(Paragraph("1. Dataset Overview", styles['Heading2']))
        overview_data = [
            ["Source File", file_name],
            ["Total Rows", str(profile.get("row_count", 0))],
            ["Total Columns", str(profile.get("column_count", 0))],
            ["Numeric Columns", str(len(profile.get("numeric_columns", [])))],
            ["Categorical Columns", str(len(profile.get("categorical_columns", [])))],
        ]
        t = Table(overview_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))
        
        # 2. Descriptive Stats
        elements.append(Paragraph("2. Numeric Summary Statistics", styles['Heading2']))
        if stats:
            stats_data = [["Column", "Min", "Max", "Mean", "Median"]]
            for col, stat in stats.items():
                mean_val = f"{stat['mean']:.2f}" if stat['mean'] is not None else "N/A"
                stats_data.append([
                    col, 
                    str(stat['min']), 
                    str(stat['max']), 
                    mean_val, 
                    str(stat['median'])
                ])
            
            t2 = Table(stats_data)
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t2)
        else:
            elements.append(Paragraph("No numeric columns available.", styles['Normal']))
            
        elements.append(Spacer(1, 12))
        
        # 3. Predictive Analytics
        if ml_result:
            elements.append(PageBreak())
            elements.append(Paragraph("3. Predictive Analytics", styles['Heading2']))
            
            elements.append(Paragraph(f"Target Column: {ml_result['target']} ({ml_result['type'].upper()})", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            elements.append(Paragraph(f"Winning Model: {ml_result['best_model_name']}", styles['Heading3']))
            
            metrics_data = [["Metric", "Value"]]
            for k, v in ml_result["best_metrics"].items():
                metrics_data.append([k.upper(), f"{v:.4f}"])
                
            t3 = Table(metrics_data)
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t3)
            
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
