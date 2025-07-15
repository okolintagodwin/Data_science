1. Scenario
As a data scientist collaborating with a thoracic surgery center, I propose a machine learning solution to support clinical decision-making for lung cancer patients undergoing major surgery. Based on a retrospective dataset of 470 patients treated between 2007 and 2011 at the Wroclaw Thoracic Surgery Centre, I aim to develop a classification model that predicts a patient's likelihood of surviving more than one year after lung surgery.

The goal is to use historical patient data—including clinical, demographic, and operative features—to identify high-risk patients preoperatively. This predictive tool would be integrated into the hospital’s decision support system, providing clinicians with an additional layer of insight to help them:

Tailor treatment plans

Consider alternative therapies or closer postoperative monitoring for high-risk patients

Allocate resources more efficiently

2. Value of Solving the Problem
The predictive model could provide substantial clinical and operational value, such as:

Improved Patient Outcomes: Early identification of high-risk patients enables more proactive care and improved survival chances through preventive measures.

Optimized Use of Hospital Resources: By knowing which patients are at greater risk, hospitals can plan for more intensive postoperative support (e.g., ICU beds, follow-ups).

Cost Reduction: Preventing complications or readmissions for high-risk patients can lead to significant healthcare cost savings.

Enhanced Decision-Making: Surgeons and oncologists can use the model to better assess surgical risk and patient suitability, potentially reducing unnecessary surgeries for extremely high-risk patients.

In a broader context, this kind of predictive analytics could also be extended to national cancer registries or public health decision-making, supporting evidence-based policy on surgical interventions.

3. Quality Criteria
To assess the performance of the predictive models, I will use the following two quality criteria:

1. Balanced Accuracy

Why? In medical datasets, class imbalance is common. Here, if far more patients survive than die within one year, a naïve model might achieve high standard accuracy by mostly predicting survival. Balanced Accuracy accounts for this imbalance by averaging recall for each class.

Goal: Ensure that the model performs well for both classes (death and survival), not just the majority.

2. F1 Score (Harmonic Mean of Precision and Recall)

Why? F1 Score balances both precision (how many predicted positives are actually positive) and recall (how many actual positives are captured by the model). It's particularly useful when dealing with imbalanced datasets, where accuracy can be misleading.

Goal: Select a model that not only identifies most of the true positive cases (high recall), but also minimizes false positives (high precision), ensuring reliable clinical decisions when predicting post-operative mortality risk.