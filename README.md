# Santander-Product-Recommendation

This repo is for the following Kaggle competition:
https://www.kaggle.com/c/santander-product-recommendation

Contributers: Shubham Nath, Kumud Kujur and Priyanka Nath


Rank: Top 4%

Santander Solution Steps :-
1. Problem
    We were given periodical data from Jan'15 to May'16 of customers and products they have bought or used during every month. Our task is to predict what new product whether a customer will buy or use in the month of Jun'16 which he would not have done in the month of May'16 (look into kaggle page for more details)
    Out of 24 products given we've to predict top 7 product a customer will buy or use which he hasn't bought or used in the month of may'16. The ranking of these product will is necessary as with correct order we'll get higher score (See the evaluation metrics from kaggle)

2. Defining/Choosing Training Set


    2.a) We created monthly distribution of new products bought or used for each month (which was not bought/used earlier). We saw there's a clear seasonalilty when comparing previous Jun'15 to all the other months. Few products (like renta) was brought highly during this month. Also, for few products usage/purchase declined or increased during this month.
    
    
    2.b) We also saw some trends w.r.t current months (May'16, Apr'16, Mar'16). Some products continuously declined or increased w.r.t their purchase/usage from start.
    
    
    2.c) Final data - We tested these training sets separately and combined. Best individual training set came out to Jun'15 (seasonal data). And best combined data(best overall) came when used Jun'15(seasonal) & Apr'16(trend) data. This was our final training set. Also we observed certain products were not getting sold/used lately (as per trends) so we removed our target from 24 products to 21 products. Hence, we would only predict 7 top items from list of 21 products which were not bought/used earlier

3. Feature exploration/engineering
    
    3.1) Previous bought products
    Created different variables such as last 5 months products used/purchased by customer (why only last 5 month? - Because for training set jan'15, data is from Jan'15 and not beyond that). So it created 5 new features (containing products used/purchase in last 5 months) Also, other features using previous bought products inlcuded - total number of products bought/used in last 5months etc.
    
    
    3.2) Recalculating tenure (join date - purchase/use date). Recalculating as given value was not correct

  
    3.3) Outliers were capped at 99.9% values
  
  
    3.4) Missing for categorical variables
        3.4a) For variables which had lesser class values (<=5), we replaced by mode(highest frequent coming class)
        3.4b) For variables which had more class values (like channel), we binned lesser frequent classes. For eg, 'KHM' or 'KHP' to 'KH'. 'ABC' and 'AXZ' to 'A' etc
      
      
    3.5) Missing for numerics
        It was calulated on one variable (household income). For the missing ones we saw combination of other variables such as 'Customer Segment' & 'Province/District' and imputed with Mean value of this combo. If still some were left we imputed Mean of 'Customer Segment'


