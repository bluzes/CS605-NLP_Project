# Natural Language Generation (NLG) for Earning Calls Transcript Information extraction
- Earnings conference calls offer rich information about the company’s management views on key performance drivers, macroeconomic and future outlooks. 
- Professionals can gain valuable insights to understand the overall economic performance and outlook.
- However it is time consuming to extract insights due to large amount of text present in an earning call transcript and the large number of public traded companies
- This seeks to speed up information extraction by summarizing the document, generate key topics and sentiments
# Project Directory Structure
```
├── datasets/preloaded/small          <- datasets used
    ├── 2021_Q4_small_clean.csv
    ├── 2022_Q1_small_clean.csv
├── saved_index/preloaded             <- indexes for loading
    ├── docstore.json                 <- docstore for loading
    ├── graph_store.json              <- graph store for loading
    ├── index_store.json              <- index store for loading
    ├── vector_store.json             <- vector store for loading
├── app.py                            <- code for UI
├── requirements.txt                  <- requirements
```
# Document summary using NLG
![image](https://github.com/bluzes/CS605-NLP_Project/assets/139099352/ac6bc1ca-4164-4ca7-94ea-6aa669e27ed6)
# Topic generation using NLG
![image](https://github.com/bluzes/CS605-NLP_Project/assets/139099352/e8c9e81d-9386-4309-8543-0c6b0b9280ef)

