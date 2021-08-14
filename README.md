# ML4PPM
Machine learning models for predictive process monitoring

Predictive business process monitoring (PBPM) aims to predict future process behavior during ongoing process executions based on event log data. Especially, techniques for the next activity and timestamp prediction can help to improve the performance of operational business processes. Recently, many PBPM solutions based on deep learning were proposed by researchers. Due to the sequential nature of event log data, a common choice is to apply recurrent neural networks with long short-term memory (LSTM) cells. We argue, that the elapsed time between events is informative. However, current PBPM techniques mainly use “vanilla” LSTM cells and hand-crafted time-related control flow features. To better model the time dependencies between events, we propose a new PBPM technique based on time-aware LSTM (T-LSTM) cells. T-LSTM cells incorporate the elapsed time between consecutive events inherently to adjust the cell memory. Furthermore, we introduce cost- sensitive learning to account for the common class imbalance in event logs. Our experiments on publicly available benchmark event logs indicate the effectiveness of the introduced techniques.

## **Datasets**

**Helpdesk**: This event log originates from a ticket management process of an Italian software company.

**BPI’12 W Subprocess** (BPI12W): The Business Process Intelligence (BPI) 2012 challenge provided this event log from a German financial institution. The data come from a loan application process. The ‘W’ indicates state of the work item for the application.

**BPI'12** (BPI12): Event log provided by the Business Process Intelligence (BPI) 2012 challenge.

**BPI'17** (BPI17): Event log provided by the Business Process Intelligence (BPI) 2017 challenge.

## **Evaluation**

**Accuracy**: Evaluation of next activity prediction.

**Mean absolute error (MAE)**: Evaluation of next timestamp prediction.

## **Links**

NLP resources: <https://lena-voita.github.io/nlp_course.html>

Attention for Language Modeling: <https://www.aclweb.org/anthology/I17-1045.pdf>

Attention is All you need: <https://arxiv.org/abs/1706.03762>

Time-aware LSTM for PBPM: <https://arxiv.org/abs/2010.00889>

GANs for PBPM (2020): <https://arxiv.org/abs/2003.11268v2>

Another GANs for PBPM (2021): <https://arxiv.org/abs/2102.07298>

Focal loss: <https://arxiv.org/abs/1708.02002>
