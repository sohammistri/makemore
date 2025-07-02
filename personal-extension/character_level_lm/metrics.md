# Final Metrics of Sequential Models
| Model | Train Loss | Val Loss | Test Loss |
|------------|------------|------------|------------|
| RNN (scratch) | 1.7635 | 1.9782 | 1.9806 |
| **LSTM (scratch)** | **1.7102** | **1.9346** | **1.9395** |
| GRU (scratch) | 1.7706 | 1.9376 | 1.9404 |
| LSTM (finetuned, 118 vocab) | 1.7298 | 2.0069 | 2.0094 |
| GRU (finetuned, 118 vocab) | 1.9514 | 2.0541 | 2.0527 |
| LSTM (finetuned, 30 vocab) | 1.6834 | 2.0212 | 2.0226 |
| GRU (finetuned, 30 vocab) | 1.8622 | 2.0368 | 2.0405 |
