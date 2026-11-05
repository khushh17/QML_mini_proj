"""Build the final report in the supplied academic section order; no retraining."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/Final_Quantum_Computing_Project_Report.pdf'
TMP = ROOT / 'tmp/pdfs/final_report'
TMP.mkdir(parents=True, exist_ok=True)
rows = json.loads((ROOT / 'results/main/summary.json').read_text())
records = json.loads((ROOT / 'results/main/training_records.json').read_text())
RED = colors.HexColor('#A71919')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Main', fontName='Helvetica-Bold', fontSize=18, leading=23, textColor=RED, spaceAfter=18))
styles.add(ParagraphStyle(name='Sub', fontName='Helvetica-Bold', fontSize=11.5, leading=15, spaceBefore=13, spaceAfter=7))
styles.add(ParagraphStyle(name='Text', fontName='Helvetica', fontSize=10.3, leading=15, spaceAfter=9))
styles.add(ParagraphStyle(name='SmallText', fontName='Helvetica', fontSize=8.5, leading=12, spaceAfter=7))
styles.add(ParagraphStyle(name='Cell', fontName='Helvetica', fontSize=9, leading=12))
styles.add(ParagraphStyle(name='CodeText', fontName='Courier', fontSize=8, leading=12, spaceAfter=7))
styles.add(ParagraphStyle(name='Cover', fontName='Helvetica-Bold', fontSize=27, leading=33, textColor=RED, spaceAfter=22))
story = []
def p(text, style='Text'):
    story.append(Paragraph(text, styles[style]))
def sub(text): p(text, 'Sub')
def page(title):
    if story: story.append(PageBreak())
    p(title, 'Main')
def table(data, widths):
    cells = [[Paragraph(escape(str(v)), styles['Cell']) for v in row] for row in data]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('LINEABOVE',(0,0),(-1,0),1,colors.black), ('LINEBELOW',(0,0),(-1,0),.7,colors.black), ('LINEBELOW',(0,-1),(-1,-1),.7,colors.black), ('LEFTPADDING',(0,0),(-1,-1),5), ('RIGHTPADDING',(0,0),(-1,-1),5), ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(t); story.append(Spacer(1,10))
def code(text): p(escape(text), 'CodeText')

story.append(Spacer(1,65))
p('Quantum Computing<br/>Project Report', 'Cover')
p('California Housing: Classical and Quantum Regression', 'Sub')
p('A parameterized quantum circuit extension of<br/><i>Revisiting Deep Learning Models for Tabular Data</i>')
story.append(Spacer(1,24))
table([['Project focus','Continuous regression on tabular data'], ['Quantum model','Eight-qubit hybrid quantum neural network'], ['Execution','Ideal CPU simulation using PennyLane and PyTorch'], ['Experiment','Shared data split; classical baselines; saved checkpoints'], ['Report date','5 November 2026']], [120,375])
story.append(Spacer(1,22))
p('Report overview', 'Sub')
p('Abstract<br/>1. Introduction &amp; Problem Statement<br/>2. Hardware &amp; Environment Specifications<br/>3. Methodology &amp; Implementation Pipeline<br/>4. Performance &amp; Comparative Analysis<br/>5. Conclusion &amp; Future Scope<br/>References')
p('This report describes the completed local experiment. The assigned paper is classical; the quantum model is this project\'s extension. The results do not establish quantum advantage.', 'SmallText')

page('Abstract')
p('This project investigates continuous house-value prediction using classical machine learning and a hybrid quantum neural network (QNN). It uses the StatLib California Housing dataset, containing 20,640 census block-group records, and draws inspiration from the tabular ResNet architecture in <i>Revisiting Deep Learning Models for Tabular Data</i> [1]. The quantum circuit is an additional project implementation, rather than a model proposed in the paper.')
p('Eight numerical features are derived from the supplied raw data. To fit a modest CPU budget, the experiment uses 2,048 training records, 512 validation records, and 4,128 held-out test records. All models use the same selected examples. The comparison includes a mean predictor, histogram gradient-boosted trees, a compact tabular ResNet, and an eight-qubit QNN. The QNN encodes standardized inputs as rotation angles, applies trainable RY and RX gates with a CNOT ring, and converts eight Pauli-Z expectation values into a continuous prediction using a classical linear readout.')
p('The models run locally without quantum hardware. On the fixed test set, boosted trees achieve the lowest RMSE: $52,278 in historical dollars, followed by the ResNet at $58,011 and the two-layer QNN at $88,468. The QNN improves RMSE by 22.8% relative to the mean predictor, but is less accurate and slower than both trained classical baselines. The result demonstrates an operational, trainable quantum regression pipeline and the importance of measuring performance instead of assuming that quantum models are superior.')
p('<b>Keywords:</b> California Housing; tabular regression; parameterized quantum circuit; hybrid QNN; PennyLane; classical comparison.')
sub('How to read the results')
p('RMSE and MAE measure prediction error; smaller values are better. R-squared (R2) measures improvement relative to a constant prediction based on the test-target mean; larger values are better. These are regression metrics, not classification accuracy percentages. Reported dollar values refer to the historical dataset and to area-level median values.')

page('1. Introduction &amp; Problem Statement')
sub('1.1 Background')
p('Tabular data consists of rows of examples and columns of features. Here, each row describes a California census block group, and the target is its median housing value. The task is supervised regression: learn a mapping from eight numerical inputs to one continuous output. It is not a prediction of a current individual property price.')
p('The assigned paper evaluates deep learning approaches for tabular data, including ResNet and FT-Transformer [1]. This project implements a small ResNet inspired by that work and introduces a parameterized quantum circuit (PQC) as an alternative learned representation. The full paper benchmark, FT-Transformer, extensive tuning and ensembles are outside the implemented scope.')
sub('1.2 Motivation')
p('A PQC contains quantum gates whose angles can be adjusted during training. A classical optimizer updates those angles to reduce prediction error. This makes it possible to study quantum machine learning on an ordinary laptop using a simulator. A fair baseline comparison is needed to determine whether this extra complexity helps on the chosen task.')
p('<b>Problem statement:</b> How well does a shallow, eight-qubit hybrid regressor predict California Housing values compared with classical models under a fixed, low-compute experimental setup?')
sub('1.3 Objectives')
p('1. Load and validate the supplied dataset and derive meaningful numerical features.<br/>2. Implement classical baselines and a trainable quantum regression model.<br/>3. Prevent data leakage using separate training, validation and test sets.<br/>4. Compare error, training time and circuit resources using saved results.<br/>5. Provide reproducible code, gate diagrams and clear explanations for demonstration.')
sub('Dataset and feature meaning')
table([['Input','Meaning / conversion'],['MedInc; HouseAge','Median income in dataset units; median house age'],['AveRooms; AveBedrms','Total rooms / households; total bedrooms / households'],['Population; AveOccup','Block-group population; population / households'],['Latitude; Longitude','Geographical location'],['Target','Median house value divided by 100,000']], [160,335])
p('The local audit found no nonfinite values and no duplicate raw rows. Recorded targets range from $14,999 to $500,001; 965 rows (4.68%) occur at the maximum value. This ceiling limits what the target can express.', 'SmallText')

page('2. Hardware &amp; Environment Specifications')
sub('2.1 Hardware Specifications')
table([['Item','Project environment'],['Laptop specification','Intel Core i5 and 8 GB RAM, as reported by the user'],['Recorded processor identifier','Intel64 Family 6 Model 140 Stepping 1, GenuineIntel'],['Operating system','Windows 11; recorded platform build 10.0.26200'],['Compute allocation','CPU execution with two compute threads'],['GPU / quantum processor','Neither required nor used for the saved experiment'],['Quantum simulation','Eight qubits: state vector has 2^8 = 256 complex amplitudes']], [150,345])
p('The small circuit and reduced training subset keep this implementation practical on a basic laptop. Training may increase CPU usage. Memory use for the complete process was not benchmarked; framework memory and gradient calculations also contribute, so state-vector size alone is not total RAM use. Times in this report are local measurements, not guarantees for another computer.')
sub('2.2 Software &amp; Framework Stack')
table([['Software','Recorded version','Role'],['Python','3.12.14','Main implementation language'],['PennyLane','0.43.0','Quantum gates, QNode, simulator and diagrams'],['PyTorch','2.5.1+cpu','Neural layers, gradients and optimization'],['scikit-learn','1.7.2','Mean and boosted-tree baselines; metrics'],['NumPy / SciPy','2.3.4 / 1.16.2','Numerical processing'],['Matplotlib','3.10.7','Saved performance and circuit figures']], [130,100,265])
p('The simulator is PennyLane <b>default.qubit</b> with <b>shots=None</b>. It computes exact expectation values without sampling noise. The model uses the PyTorch interface and simulator backpropagation. The virtual environment and pinned dependency files support repeatable execution.')
p('The project is a command-line ML/QML implementation. A frontend is unnecessary for this demonstration because the saved models, metrics and circuit explanation can be inspected directly.')

page('3. Methodology &amp; Implementation Pipeline')
sub('3.1 Baseline Algorithm Selection')
table([['Model','Configuration and purpose'],['Mean predictor','Predicts the training-target mean; minimum useful baseline.'],['Boosted trees','Histogram gradient boosting: 200 iterations, 15 maximum leaf nodes, learning rate 0.08, L2 regularization 1.0.'],['Tabular ResNet','8 inputs -> width 32 -> two residual blocks -> scalar output; 8,897 trainable parameters.'],['Main QNN','Eight qubits, two encoding/rotation/entangling layers, linear readout; 41 trainable parameters.'],['Depth comparison','One-layer QNN with 25 parameters; one exploratory seed.']], [115,380])
p('Each ResNet block applies BatchNorm, Linear(32,64), ReLU, dropout 0.1, Linear(64,32) and dropout 0.1, then adds the block input. The residual connection helps information and gradients pass through the network. A final BatchNorm, ReLU and linear layer produce the output.')
sub('Data preparation and split')
p('The supplied <b>cal_housing.data</b> has nine raw columns described by <b>cal_housing.domain</b>. Household ratios are calculated before modeling. With split seed 42, the initial partitions contain 13,209 training-pool rows, 3,303 validation-pool rows and 4,128 test rows. Fixed selections use 2,048 training and 512 validation rows; 13,952 rows remain unused to reduce compute.')
p('Feature means and standard deviations are fitted only on the selected training rows. Neural inputs become z = (x - training mean) / training standard deviation. Neural targets are also standardized for training, then predictions are converted back before evaluation. Trees and the mean baseline use unscaled features and targets. All models see the same underlying examples.')
sub('Pipeline')
p('<b>Local data -> audit and feature conversion -> fixed split -> training-only preprocessing -> model fitting -> best validation checkpoint -> held-out test evaluation -> saved metrics, predictions and diagrams.</b>')
p('This is a comparison of complete implemented pipelines. It is not an exact paper reproduction: the paper uses a different preprocessing and tuning protocol. No test examples are used to fit scaling, update parameters or select the best training epoch.')

page('3. Methodology &amp; Implementation Pipeline')
sub('3.2 Quantum Circuit Optimization')
p('All eight qubits start in the zero state. Input feature j becomes angle a_j = 2 arctan(z_j), which keeps the angle between -pi and pi. Each layer applies RY(a_j), trainable RY(theta_lj0), and trainable RX(theta_lj1) on every wire, followed by the ordered CNOT ring 0->1, 1->2, ..., 6->7, 7->0. The main model repeats this sequence twice, re-uploading the same inputs in the second layer.')
img = PILImage.open(ROOT / 'results/main/figures/qnn_gates.png').convert('L')
img.save(TMP / 'circuit_monochrome.png')
story.append(Image(str(TMP / 'circuit_monochrome.png'), width=495, height=201.1))
p('Figure 1. Gate structure drawn from the implemented QNode; angle values are hidden. RY and RX rotate a qubit state. A CNOT uses one control wire and one target wire, allowing correlations between qubits. The drawing itself does not execute training.', 'SmallText')
p('The circuit returns eight Pauli-Z expectation values m_j, each between -1 and 1. The classical output is y_scaled = b + sum(w_j m_j). Inverse target scaling gives the reported value. There is no direct classical shortcut from the inputs to the output around the quantum circuit.')
p('Training minimizes mean squared error on standardized targets using AdamW. The QNN learning rate is 0.02; the ResNet learning rate is 0.001. Both use weight decay 0.0001 and gradient-norm clipping at 5. Gradients flow through the simulated circuit and update both circuit angles and linear readout weights.')
p('One epoch means one full pass through the training set. With 2,048 rows and batches of 64, an epoch contains 32 batches. Training allows up to 40 epochs and stops after 10 epochs without a validation-MSE improvement of at least 0.000001. The best validation checkpoint is restored. The main neural models use seeds 7, 17 and 27.')

page('3. Methodology &amp; Implementation Pipeline')
sub('3.3 Quantum Resource Optimization')
p('Resource control comes from using eight qubits, shallow circuits, a small training subset, batch processing and two CPU threads. Each feature maps directly to one qubit, so there is no separate dense input network. The two-layer circuit contains 32 learned angles; the output layer adds eight weights and one bias, for 41 trainable parameters.')
p('The one-layer model has 16 learned angles and nine output parameters. It provides an exploratory depth comparison. Fewer parameters do not automatically imply lower runtime: the simulator must propagate quantum states and gradients for every batch.')
p('The adjacent encoding RY(a) and trainable RY(b) gates on the same wire could mathematically merge into RY(a+b). The submitted implementation keeps them separate so that input encoding and learned offsets remain easy to see. No gate-fusion pass, hardware mapping or quantum error correction is claimed.')
sub('3.4 Compilation &amp; Execution Engine')
p('The circuit is defined as a PennyLane QNode in <b>housing/models.py</b>. PennyLane records the gate operations and executes them on <b>default.qubit</b>. PyTorch differentiates the ideal simulator using <b>diff_method="backprop"</b>. This is simulator execution, not compilation to a physical quantum device, and it does not use a hardware parameter-shift experiment.')
p('The training loop stores configuration, split indices, preprocessing statistics, model checkpoints, predictions and metrics under <b>results/main</b>. Diagram utilities draw the same circuit definition; the saved-model demonstration actually evaluates the model.')
sub('How to run the completed project')
p('Run these commands in PowerShell from the project folder with its existing virtual environment. Demonstration and drawing do not retrain the models.', 'SmallText')
code(r'.\.venv\Scripts\python.exe run.py demo')
code(r'.\.venv\Scripts\python.exe run.py explain')
code(r'.\.venv\Scripts\python.exe terminal_circuit.py --ascii --width 58')
code(r'.\.venv\Scripts\python.exe generate_diagrams.py')
p('For a new training run, choose a new output directory:', 'SmallText')
code(r'.\.venv\Scripts\python.exe run.py train --output results/repeat')
p('Use <b>README.md</b> for installation on another machine. Recreate the virtual environment there rather than copying this laptop\'s environment.')

page('4. Performance &amp; Comparative Analysis')
sub('4.1 Classical vs. Quantum Comparative Matrix')
names = {'mean':'Mean predictor','boosted_trees':'Boosted trees','resnet':'Tabular ResNet','qnn':'QNN, two layers','qnn_1layer':'QNN, one layer*'}
data = [['Model','Runs','RMSE ($)','MAE ($)','R2','Train (s)']]
for r in rows:
    data.append([names[r['model']],r['runs'],f"{r['rmse_mean']*100000:,.0f}",f"{r['mae_mean']*100000:,.0f}",f"{r['r2_mean']:.4f}",f"{r['train_seconds_mean']:.3f}" if r['model']=='mean' else f"{r['train_seconds_mean']:.1f}"])
table(data,[135,35,85,85,65,90])
p('Dollar errors refer to historical median values. Repeated-run entries are means. *The one-layer QNN is a single-seed exploratory comparison, not a replacement selected after inspecting test performance.', 'SmallText')
p('RMSE = square root of the mean squared prediction error. MAE = mean absolute prediction error. R2 = 1 - (sum of squared errors / sum of squared deviations from the test-target mean). RMSE gives larger errors more weight than MAE. A negative R2 is possible when predictions are worse than that reference constant.')
page('4. Performance &amp; Comparative Analysis')
sub('Test-error comparison')
chart = Drawing(495, 205)
for i, r in enumerate(rows):
    y = 175 - i * 33
    value = r['rmse_mean'] * 100000
    x = 132
    width = value / 120000 * 275
    chart.add(String(0, y + 4, names[r['model']], fontName='Helvetica', fontSize=9))
    chart.add(Rect(x, y, width, 16, fillColor=colors.white if r['model'].startswith('qnn') else colors.black, strokeColor=colors.black, strokeWidth=1))
    if r['rmse_std'] is not None:
        spread = r['rmse_std'] * 100000 / 120000 * 275
        chart.add(Line(x+width-spread, y+8, x+width+spread, y+8, strokeColor=colors.black))
        for end in (x+width-spread, x+width+spread):
            chart.add(Line(end,y+3,end,y+13,strokeColor=colors.black))
    chart.add(String(x + width + 10, y + 4, f'${value:,.0f}', fontName='Helvetica', fontSize=9))
chart.add(Line(132,30,407,30,strokeColor=colors.black))
for amount in (0,40000,80000,120000):
    x=132+amount/120000*275
    chart.add(Line(x,27,x,30,strokeColor=colors.black))
    chart.add(String(x,14,f'${amount//1000}k',fontName='Helvetica',fontSize=8,textAnchor='middle'))
story.append(chart)
p('Figure 2. Held-out test RMSE in historical dollars; lower is better. Filled bars: classical models. Outlined bars: QNNs. Whiskers show one sample standard deviation for the three-seed ResNet and two-layer QNN. *The one-layer QNN has one exploratory run.', 'SmallText')
sub('4.2 Quantum Resource Utilization')
table([['Resource','One-layer QNN','Two-layer QNN'],['Qubits / state amplitudes','8 / 256','8 / 256'],['Trainable circuit angles','16','32'],['Classical output parameters','9','9'],['Total learned parameters','25','41'],['Readout','8 Z expectations','8 Z expectations'],['Shots / device','Exact / CPU simulator','Exact / CPU simulator']], [215,140,140])
p('The expectation values are analytic, not finite-shot measurements. No hardware queue time, device calibration, connectivity overhead or peak-memory measurement was recorded. The resource counts describe the logical circuit actually defined in code.')

page('4. Performance &amp; Comparative Analysis')
sub('4.3 Execution Time &amp; Accuracy Analysis')
p('Boosted trees achieve the lowest measured test error and train in about 0.4 seconds. The ResNet averages 5.9 seconds per fit, while the two-layer QNN averages 116.9 seconds. Thus, the QNN takes about 20 times the ResNet training time in this run despite having fewer parameters. Simulator work and differentiation make parameter count alone an unreliable runtime estimate.')
data=[['Model / seed','Epochs run','Best epoch','Training seconds']]
for r in records:
    if 'epochs_run' in r: data.append([f"{names[r['model']]} / {r['seed']}",r['epochs_run'],r['best_epoch'],f"{r['train_seconds']:.2f}"])
table(data,[205,85,85,120])
p('The mean baseline does not use epochs. Boosted trees use 200 boosting iterations. All nine fits total approximately 7 minutes 23 seconds. These times include training and validation, but exclude dependency installation, final test prediction, plotting and report generation.', 'SmallText')
p('Across three seeds, test-RMSE sample standard deviation is approximately $647 for the ResNet and $1,193 for the main QNN. These describe initialization variation on one fixed split, not confidence intervals or variation across different datasets.')
sub('4.4 Circuit Depth &amp; Gate Count Analysis')
table([['Logical circuit resource','One layer','Two layers'],['Input RY gates','8','16'],['Trainable RY gates','8','16'],['Trainable RX gates','8','16'],['CNOT gates','8','16'],['Total gates (excluding readout)','32','64'],['Logical depth (unmerged gates)','11','22']], [275,110,110])
p('Each layer has three parallel rounds of single-qubit rotations and an ordered chain of eight dependent CNOT gates. Depth counts the longest gate-dependency path, not simply the number of Python loops. Measurements are excluded. These counts precede any hardware decomposition or gate merging.', 'SmallText')

page('4. Performance &amp; Comparative Analysis')
sub('4.5 Noise Impact &amp; Error Analysis')
p('<b>No noise experiment was performed.</b> The simulator uses exact expectation values with no finite-shot sampling, gate noise or readout noise. The reported prediction errors therefore measure limitations of the learned regression pipeline, not errors measured on a quantum processor. The experiment cannot establish noise robustness.')
p('Possible contributors to prediction error include the small training subset, fixed model settings, limited circuit expressiveness, and compression of large standardized inputs by the arctangent encoding. These are plausible explanations, not individually established causes. The difference between RMSE and MAE indicates that larger prediction errors affect the aggregate RMSE substantially.')
p('The maximum-value ceiling in the dataset also limits target resolution. Random geographical splits may place nearby block groups in both training and test sets; the measured performance does not establish generalization to entirely new regions or present-day housing prices.')
sub('4.6 Experimental Results &amp; Discussion')
p('The two-layer QNN reduces RMSE by 22.8% relative to the mean predictor, showing that this implemented pipeline learns useful predictive structure. However, boosted trees and the ResNet achieve lower error with less training time. The evidence supports quantum regression feasibility on a CPU simulator, not quantum advantage.')
p('The one-layer QNN has RMSE $108,619, compared with $88,468 for the mean of the two-layer QNN runs. This suggests that the selected deeper circuit is more effective in this experiment, but the one-layer comparison uses only one seed. It does not establish a statistically reliable depth effect.')
p('The saved best-checkpoint circuit angles differ from their initial values by L2 distances 5.0030, 4.8408 and 6.6711 for the main QNN seeds. This confirms that the quantum parameters changed during training. Parameter movement alone is not proof of useful learning; held-out comparison supplies the separate performance evidence.')
sub('Verification and reproducibility')
p('The saved verification record reports eight passing tests, reproduced checkpoint predictions and reproduced metrics for all nine runs at a comparison tolerance of 1e-10. Checks cover feature conversion, disjoint splits, training-only preprocessing, quantum gradients and updates, and model save/load behavior. Split indices, preprocessing statistics and checkpoints are retained for inspection.')
p('No hyperparameter search or equal-wall-time tuning was performed. Model capacities and preprocessing differ. Conclusions apply to these fixed pipelines and this split, rather than to all classical or quantum models.')

page('5. Conclusion &amp; Future Scope')
sub('5.1 Summary of Results')
p('The project successfully implements an eight-qubit, trainable hybrid quantum regressor for California Housing and compares it with classical baselines using shared data. Boosted trees perform best. The QNN beats a constant mean prediction but is slower and less accurate than the trained classical models. The main learning outcome is understanding the complete path from tabular features through quantum encoding, circuit optimization and measurement to a continuous prediction, with evidence-based evaluation.')
p('<b>Short presentation explanation:</b> We extended a classical tabular deep learning study with a simulated quantum neural network. Eight housing features are encoded as rotation angles on eight qubits. Trainable RY/RX rotations and CNOT gates transform the state, and eight Z expectations feed a linear output. We trained and compared the models on the same data. The QNN learned, but boosted trees achieved the best accuracy and runtime in our experiment.')
sub('5.2 Future Improvements')
p('Increase the training subset when compute permits; tune models using validation data with a declared budget; repeat the one-layer comparison across multiple seeds; and test several data splits, including geographical holdouts. Compare encodings and entangling layouts through controlled experiments. Add finite-shot and explicit noise simulations before making claims about hardware suitability. Consider hardware execution only after documenting gate mapping and measurement costs.')
p('Further changes should be selected using validation results, with a fresh or strictly protected final test evaluation. Repeatedly redesigning models based on the same test set would weaken the reliability of the reported results.')
sub('5.3 GitHub Repository Link')
p('<link href="https://github.com/khushh17/QML_mini_proj" color="black">https://github.com/khushh17/QML_mini_proj</link>')
p('Project repository supplied by the team. It is separate from the paper authors\' reference implementation listed in the references.')

page('References')
refs = [
('1','Y. Gorishniy, I. Rubachev, V. Khrulkov and A. Babenko. Revisiting Deep Learning Models for Tabular Data. NeurIPS, 2021. Supplied local PDF: references/2106.11959v5.pdf.','https://arxiv.org/abs/2106.11959'),
('2','Paper authors\' reference implementation. This is the source paper repository, not the student project repository.','https://github.com/yandex-research/tabular-dl-revisiting-models'),
('3','StatLib California Housing dataset, supplied locally as CaliforniaHousing/cal_housing.data and cal_housing.domain. Dataset archive mirror:','https://figshare.com/articles/dataset/cal_housing_tgz/3829992'),
('4','Scikit-learn California Housing loader source: feature conversion and target scaling.','https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_california_housing.py'),
('5','PennyLane documentation: PyTorch interface and differentiable quantum models.','https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html')]
for n, desc, url in refs:
    p(f'[{n}] {escape(desc)}')
    p(f'<link href="{url}" color="black">{url}</link>', 'SmallText')
sub('Local implementation and experiment evidence')
p('<b>housing/models.py:</b> actual ResNet and QNN definitions.<br/><b>housing/data.py:</b> data conversion and preprocessing.<br/><b>housing/experiment.py:</b> training and evaluation.<br/><b>results/main/config.json:</b> environment and experiment settings.<br/><b>results/main/summary.json:</b> aggregate test metrics.<br/><b>results/main/training_records.json:</b> epochs, timing and parameter changes.<br/><b>results/main/verification.json:</b> saved correctness checks.<br/><b>generate_diagrams.py / terminal_circuit.py:</b> circuit visualization utilities.', 'SmallText')
p('Numerical claims in this report come from the completed local run, not from the source paper\'s benchmark tables. External links identify background sources; no new benchmark or hardware experiment was run while preparing this report.', 'SmallText')

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(.4)
    canvas.line(50, 42, A4[0]-50, 42)
    canvas.setFillColor(colors.black)
    canvas.setFont('Helvetica',8)
    canvas.drawString(50,29,'California Housing | Classical and Quantum Regression')
    canvas.drawRightString(A4[0]-50,29,str(doc.page))
    canvas.restoreState()

OUT.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=48,bottomMargin=57,title='Quantum Computing Project Report: California Housing',author='California Housing Project Team')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
