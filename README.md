# Learn to Rank (LETOR) for ACA Project

###Components:
- **Feature extraction:** data preprocessing to extract feature from MongoDB
- **Offline training:** train LETOR model based on user click through data, using extracted features
- **Online prediction:** called by ElasticSearch to incorporate LETOR ranking with local search results

![alt tag](https://github.com/leiyang-mids/aca_letor/blob/master/ACA%20LETOR%20Data%20Flow.png)
