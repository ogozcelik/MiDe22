# MiDe22
Official data repository of English and Turkish misinformation detection datasets from the LREC-COLING 2024 paper "MiDe22: An Annotated Multi-Event Tweet Dataset for Misinformation Detection".

![Screenshot](supplementary/all_events.png)

# Dataset
The dataset comprises 10,348 tweets: 5,284 for English and 5,064 for Turkish. Tweets in the dataset cover different topics: the Russia-Ukraine war, the COVID-19 pandemic, Refugees, and additional miscellaneous events. Three misinformation labels of the tweet are also given. Since we follow Twitter's Terms and Conditions, we publish tweet IDs, not the tweet content directly. Explanations of the columns of the file are as follows:

| Column Name  | Description |
| ------------- | ------------- |
| Topic | Topic of the tweet: Ukraine, Covid, Refugees or Misc |
| Event | Event of the tweet: EN01-EN40 in English and TR01-TR40 in Turkish |
| Label | Label of the tweet: True, False, or Other|
| Tweet_id | Twitter ID of the tweet|

The distribution of tweet counts in the dataset is as follows:

| Lang | Topic | True | False | Other | Total |
|----------|----------|----------|----------|----------|----------|
| EN | Ukraine<br>Covid<br>Refugees<br>Misc<br><b>Total</b> | 320<br>167<br>94<br>146<br><b>727</b> | 393<br>514<br>328<br>494<br><b>1,729 | 618<br>663<br>796<br>751<br><b>2,828 | 1,331<br>1,344<br>1,218<br>1,391<br><b>5,284
| TR | Ukraine<br>Covid<br>Refugees<br>Misc<br><b>Total</b> | 129<br>190<br>61<br>289<br><b>669 | 338<br>558<br>202<br>634<br><b>1,732 | 477<br>816<br>298<br>1,072<br><b>2,663 | 944<br>1,564<br>561<br>1,995<br><b>5,064

## Citation
If you make use of the datasets and codes, please cite the following paper:
```bibtex
@inproceedings{toraman-etal-2024-mide22,
    title = "{M}i{D}e22: An Annotated Multi-Event Tweet Dataset for Misinformation Detection",
    author = "Toraman, Cagri  and
      Ozcelik, Oguzhan  and
      Sahinuc, Furkan  and
      Can, Fazli",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.986",
    pages = "11283--11295"}
```
