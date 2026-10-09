"""Create a plain one-page project summary with one text style."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
metrics = json.loads((root / 'reports/metrics.json').read_text(encoding='utf-8'))
scores = metrics['scores']
output = root / 'output/pdf/news_summarization_project_summary.pdf'
output.parent.mkdir(parents=True, exist_ok=True)

paragraphs = [
    'The News Summarization Service was developed to turn long English news articles into shorter summaries that are easier to read. A user can paste an article into the service and receive a summary of its main points. The project focuses on using an existing language model, building a working API around it, and checking how well its summaries match those written by people.',
    'The model used is DistilBART CNN from Hugging Face, which is already trained for news summarization. It runs locally using Python and PyTorch, with the dependencies installed in a Conda environment. Before an article reaches the model, the text is cleaned to remove unnecessary spaces and invisible characters while keeping its punctuation and capitalization. Articles longer than 1,024 model tokens are shortened to fit the input limit, and the response tells the user when this happens. No additional training or fine-tuning was carried out.',
    'FastAPI provides the interface for submitting articles and receiving summaries. The response includes the generated text, token counts, processing time, and whether the article was truncated. The API also has an interactive documentation page where users can try an example or enter their own article. The same source code is used by the API, the evaluation program, and the demonstration notebook.',
    f"The service was evaluated on 500 randomly selected articles from the official CNN/DailyMail 3.0.0 test split. A fixed random seed was used so that the same sample can be selected again. Each generated summary was compared with the article's human-written highlights using ROUGE. On a scale of 0 to 100, the average F1 scores were {scores['rouge1']['mean_f1']:.2f} for ROUGE-1, {scores['rouge2']['mean_f1']:.2f} for ROUGE-2, and {scores['rougeL']['mean_f1']:.2f} for ROUGE-L. Average processing time was about {metrics['latency_seconds']['mean']:.2f} seconds per article on an RTX 4060 Laptop GPU after the model had loaded.",
    'The completed project includes the source code, an executed notebook, sample articles and summaries, evaluation results, and setup instructions. All 25 tests passed, and the eight notebook code cells ran without errors. The service was checked on both GPU and CPU, and requests to the running API were tested. Dataset and model versions, package versions, and evaluation settings were recorded to make the work reproducible. Development was saved through local Git commits.',
    'There are still limits to the summaries. In the evaluation sample, 142 articles, or 28.4%, exceeded the input limit, so information near the end could be lost. Reviewing sample outputs also revealed omitted details, a wrong team attribution, and an unfinished sentence. ROUGE measures overlap with a reference summary rather than factual accuracy, so generated claims still need to be checked against the original article. The project demonstrates a working local summarization service; URL scraping, multilingual support, and production deployment were outside its scope.'
]

style = ParagraphStyle('Plain', fontName='Times-Roman', fontSize=12, leading=16,
                       spaceAfter=12, textColor='black')
doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=56, rightMargin=56,
                        topMargin=56, bottomMargin=56,
                        title='News Summarization Service - Project Summary',
                        author='News Summarization Service Project')
doc.build([Paragraph(text, style) for text in paragraphs])
reader = PdfReader(output)
assert len(reader.pages) == 1, f'Expected one page; found {len(reader.pages)}'
print(f'Saved plain one-page PDF: {output}')
print(f'Word count: {sum(len(p.split()) for p in paragraphs)}')
