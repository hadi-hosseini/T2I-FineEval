import json
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, 'Results', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def create_table(self, data):
        self.set_font('Arial', 'B', 10)
        col_widths = [60, 40, 40, 40] 
        row_height = self.font_size * 1.5

        for col_width, header in zip(col_widths, data[0]):
            self.cell(col_width, row_height, str(header), border=1, align='C')
        self.ln(row_height)

        self.set_font('Arial', '', 10)
        for row in data[1:]:
            if row[1] == "finegrained_vqa_score" or row[0] == "finegrained_vqa_score":
                self.set_font('Arial', 'B', 10)
            else:
                self.set_font('Arial', '', 10)
            for col_width, datum in zip(col_widths, row):
                self.cell(col_width, row_height, str(datum), border=1, align='C')
            self.ln(row_height)




def calculate_correlation(json_file_path, csv_file_path, pdf_output_path): 
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    image2category = dict()
    for entity in data:
        category = entity['category']
        image_name = entity['image_name']
        image2category[image_name] = category

    scores_df = pd.read_csv(csv_file_path)

    category_scores = []
    for _, row in scores_df.iterrows():
        image_name = row['image_name']
        category = image2category[image_name]
        category_scores.append(category)

    scores_df['category'] = category_scores


    metrics = ['clip_score', 'dascore', 'finegrained_vqa_score', 'tifa_score', 'DSG']
    category_result = []
    results = []

    for category, group_df in scores_df.groupby('category'):
        for metric in metrics:
            kendall_tau_corr, _ = kendalltau(group_df[metric], group_df['human_score'])
            spearman_rho_corr, _ = spearmanr(group_df[metric], group_df['human_score'])
            category_result.append({
                'Category': category,
                'Metric': metric,
                "Kendall's tau": round(kendall_tau_corr, 2),
                "Spearman's rho": round(spearman_rho_corr, 2)
            })

    for metric in metrics:
        kendall_tau_corr, _ = kendalltau(scores_df[metric], scores_df['human_score'])
        spearman_rho_corr, _ = spearmanr(scores_df[metric], scores_df['human_score'])
        results.append({
            'Metric': metric,
            "Kendall's tau": round(kendall_tau_corr, 2),
            "Spearman's rho": round(spearman_rho_corr, 2)
        })

    results_df = pd.DataFrame(results)
    category_results_df = pd.DataFrame(category_result)
    
    
    category_table_data = [["Category", "Metric", "Kendall's tau", "Spearman's rho"]]
    for _, row in category_results_df.iterrows():
        category_table_data.append([row['Category'], row['Metric'], row["Kendall's tau"], row["Spearman's rho"]])

    table_data = [["Metric", "Kendall's tau", "Spearman's rho"]]
    for _, row in results_df.iterrows():
        table_data.append([row['Metric'], row["Kendall's tau"], row["Spearman's rho"]])

    pdf = PDF()
    pdf.add_page()
    pdf.create_table(category_table_data)
    pdf.add_page()
    pdf.create_table(table_data)
    pdf.output(pdf_output_path)




def calculate_correlation_without_category(csv_file_path, pdf_output_path):
    scores_df = pd.read_csv(csv_file_path)

    results = []
    metrics = ['dascore', 'tifa', 'finegrained_vqa_score', 'DSG']
    for metric in metrics:
        kendall_tau_corr, _ = kendalltau(scores_df[metric], scores_df['human_score'])
        spearman_rho_corr, _ = spearmanr(scores_df[metric], scores_df['human_score'])
        results.append({
            'Metric': metric,
            "Kendall's tau": round(kendall_tau_corr, 2),
            "Spearman's rho": round(spearman_rho_corr, 2)
        })

    results_df = pd.DataFrame(results)

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 10, 'Results', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(10)

        def create_table(self, data):
            self.set_font('Arial', 'B', 10)
            col_widths = [60, 40, 40, 40] 
            row_height = self.font_size * 1.5

            for col_width, header in zip(col_widths, data[0]):
                self.cell(col_width, row_height, str(header), border=1, align='C')
            self.ln(row_height)

            self.set_font('Arial', '', 10)
            for row in data[1:]:
                if row[1] == "finegrained_vqa_score" or row[0] == "finegrained_vqa_score":
                    self.set_font('Arial', 'B', 10)
                else:
                    self.set_font('Arial', '', 10)
                for col_width, datum in zip(col_widths, row):
                    self.cell(col_width, row_height, str(datum), border=1, align='C')
                self.ln(row_height)


    table_data = [["Metric", "Kendall's tau", "Spearman's rho"]]
    for index, row in results_df.iterrows():
        table_data.append([row['Metric'], row["Kendall's tau"], row["Spearman's rho"]])

    pdf = PDF()
    pdf.add_page()
    pdf.create_table(table_data)
    pdf.output(pdf_output_path)
