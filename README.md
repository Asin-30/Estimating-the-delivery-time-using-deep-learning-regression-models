# Estimating-the-delivery-time-using-deep-learning-regression-models
This case study compares the performance of algorithms from classical machine learning like Bagging (Random Forest) and Deep Learning model.

## Objective
An online food delivery app has a number of delivery  partners available for delivering the food, from various restaurants and wants to get an estimated delivery time that it can provide the customers on the basis of what they are ordering, from where and also the delivery partners.
<br>
This dataset has the required data to train a regression model that will do the delivery time estimation, based on all those features.

### Data Profiling
Each row in this file corresponds to one unique delivery. Each column corresponds to a feature as explained below.

- **market_id** : integer id for the market where the restaurant lies
- **created_at** : the timestamp at which the order was placed
- **actual_delivery_time** : the timestamp when the order was delivered
- **store_primary_category** : category for the restaurant
- **order_protocol** : integer code value for order protocol(how the order was placed ie: through porter, call to restaurant, pre booked, third part etc)
- **total_items subtotal** : final price of the order
- **num_distinct_items** : the number of distinct items in the order
- **min_item_price** : price of the cheapest item in the order
- **max_item_price** : price of the costliest item in order
- **total_onshift_partners** : number of delivery partners on duty at the time order was placed
- **total_busy_partners** : number of delivery partners attending to other tasks
- **total_outstanding_orders** : total number of orders to be fulfilled at the moment

## Understanding data
**Structure**
- 197428 records, 14 features
- memory usage : 21.1+ MB

**data types**
- 5 features of float64 datatype
- 4 features of object datatype
- 5 features of int6 datatype

**Other**
- Max null values are for feature: *total_onshift_partners*, *total_busy_Partners*, *total_outstanding_orders* all 8.237% and *store_primary_category* with 2.411% and *market_id* with 0.5%.
- No duplicate values are present

### Data Preprocessing
- Change the feature with dates into datetime data type.
- drop the records with no delivery time.

### Feature Engineering
- Create new feature *time_taken* which is the difference of order time (*created_at*) and delivery time (*actual_delivery_time*).
- Extract the month, year, day, week, hour, minute like data from these above mentioned features to create new features out of it.

### Data Cleaning
- Replace the missing values with **mode** for categorical and **median** for numerical.
- With the help of target encoding, encode the features: *store_id* and *store_primary_category*.
- Finally, drop the *created_at* and *actual_delivery_time* from the dataset.

### Exploratory Data Analysis (EDA)
**1. % of orders as per different features** <br>
<br>
  ![image](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/62153d5c-43d1-41d8-b857-5512a7bb8f52)
  <br>
  <br>
  *Observation*
- Most of the order (27.87%) were for the market_id: 2 and least (7.32)% came for market_id: 6.
- All the orders were placed in the year 2015.
- All the orders were placed in the month of January and February. February being the month for maximum orders.
- As it is very likely, month of delivery follows the same patter as of the order placements.
  
**2. Which are the top 10 and bottom 10 stores where is business is best/worst ?** <br>
<br>
![Best and worst performing stores](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/54081cf3-71fd-42d2-955e-9d5e512282f4)


**3. What are the store's primary categories of selling that are best/worst for business?** <br>
<br>
![Best   Worst performing food categories](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/db5264bf-bbeb-4fc9-a026-a21f674812ff)

**4. How the delivery time is varying with price and type of food being ordered?** <br>
<br>
![Distribution of numerical columns](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/0662bb89-3527-4b18-8706-5dd7bb361ea6)

*Observation*
- No matter the store category, top 10 food categories takes almost same time.
- Almost all the food categories's subtotal is concentrated in a speicific sub-total ie.e 0 - 15000 dollars.
- Mean and median time taken to deliver the orders by the stores in these top 10 food categories are 47.7 and 44.28 minutes respectively.

**5. How the different prices is varying with delivery time?** <br>
<br>
![time_taken vs different prices](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/4ea9b163-1a04-4d33-8159-fa7a730153c5)
<br>
And after removing the extreme cases.
<br>
![Time_taken vs subtotal, min_item_price, max_item_price](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/e8851450-12f4-4cd1-99db-3adfe68fc862)

<br>


**6. Comparing the performance of different markets based on the available features.** <br>
<br>
![market_id vs other features](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/cecd3cea-413f-454f-9000-a026fd0b6cca)

*Observation*
- Market with id = 2 has the max number of stores followed by market = 4. Market 6 has least number of stores.
- Stores in market 2 takes least time to deliver their food and market 1 takes the most time. However, it should be noted that there is not very significant difference.
- Total items ordered are maximum for the stores located in market 4 followed by 1.
- Subtotal is max for market 4 followed by 1 and least is for 5.
- Total number of distinct items ordered were from the market 4.
- Market 3 has the products with max price.
- TOtal onshift partners are max for market 2 and followed by 4 and is least for market 3
- TOtal busy partners are max for market 4 and 2 and least for market 3.
  


**7. Comparing the performance of different food categories based on the available features.** <br>

*Observation*
- Maximum number of stores are for 'american' food, followed by 'pizza' and 'mexican' categories. Least number of stores are for *fast* and *Indian*, among the popular food categories.
- Time takes to deliver food is highest among japanese stores, followed by pizza and indian food stores.
- Total number of items offered is max for the category of *fast* and least for *pizza*.
- *Japanese* food comes first in terms of business followed by *pizza* and *indian*.
- In terms of number of items being ordered *Japanese* and *Indians* stores are at the top.
- Most expensive food items belong to the category of *pizza* and *american*. While *fast* and *Indian* have the least expensive items available among all the top 10 food categories.
- Total onshift partners are max for *Indian* restaruants and mininmum for *fast* stores.
- TOtal busy partners are max for *Indian* and least for *fast*.



**8. what food category stores are most prominent in each market?** <br>
<br>
5 of the 6 markets have store having *american* as their primary food category.


**9. Variation of sales on each day of month**<br>
<br>
![Mean_sales on each day of month](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/2514b1be-69a2-4ee1-96e7-0eecb8c489bc)



**10. variation of sales with each hour in a day**<br>
<br>
It appears that people like to order food more at late night.
![Mean sales on each hour of day](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/233f5d72-f059-4436-a592-f29d98b002c3)


**11. Variation in sales in  a week**<br>
<br>
- On weekends people like to order food relatively more as compare to other days.
![Mean sales of each weekday](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/5b9e8b42-6092-462c-b48e-035b2a39db2b)

<br>

**12. Correlation among features**<br>
![Heatmap for Correlation ](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/172139e2-ec0b-4ad2-9407-852db7d3f3b1)

**Observation**
- The delivery time is not showing any correlation with any of the features.
- *Subtotal* is showing a slight positive correlation with *num_distinct_items*.
- *total_onshift_partners* are strongly positively correlated with *total_outstanding_orders* and *total_busy_partners*.
- *total_busy_partners* are strongly positively correlated with *total_outstanding_partners* and *total_onshift_parnters*.

### Outlier Detection and Handling
IQR methods is used for detection and handling.
**Detection**<br>
<br>
![Outliers Detection using distplot and boxplot](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/7da2bf0d-3f1f-407d-aa94-f210ed228493)

**Outlier Removal**<br>
<br>
There was 3.18% of drop in records after removing outliers from *time_taken* and over all 7.53% drop of rows after dropping outliers from *time_taken* and *sub_total* both.

Post outlier removal features looked slightly normal on comparison to the earlier ones.<br>
![Distribution after outlier removal](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/ea5bd2a4-dfd1-4a80-8c1f-bd21a0244611)

<br>



### Train and Test split
Next, we will split the data. Test data will be 20% of the whole dataset.

### Modelling
#### 1. Random Forest Regression
We will use `sklearn.ensemble` to build this model. We will first start with basic model and later hyper tune to improve its performance.
Upon evaluating the untuned this model by calculating the errors:
- Mean Absolute Error: 3.1842367785142986
- Mean Squared Error: 32.55313082977181
- Root Mean Squared Error: 5.70553510459552

For the hypertuned model we got:
- Mean Absolute Error: 3.275858205735729
- Mean Squared Error: 34.017430883926515
- Root Mean Squared Error: 5.832446389288676

There is not much difference in the performance of the model even after hyperparametric tuning. Also, one can observe that classical ML model are not performing well with this dataset. Next, we  will see how neural network perform for this dataset.

#### 2 Neural Network
**2.1 Scaling**<br>
Before moving ahead, we need to scale our data first. We will do normalization on our dataset and use sklearn's `MinMaxScaler` for the same.
<br>
<br>
**2.2 Building and training the model**<br>
For the sake of easy of computation we will use only one hidden layer  in our Neural network.
<br>
for our first model we use 2/3 of total rows in train dataset as neurons present in hidden layer. We will use callback function `EarlyStopping` to stop the training when training and validation loss attain a flat curve.
We will also use BatchNormalization, regularization and dropout to prevent overfitting. We will train the model for 30 epochs.
![Train and valid loss for NN1](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/f2d49eab-e02e-4b71-bcc7-ea71764c77a0)


**2.3 Tuning the NN model**<br>
In this model we will decrease number of neurons in the hidden layers by a little.  Introduce a callback `LearningRateScheduler`, this will change the learning rate at every epoch and apply that learning rate to optimizer  as training progresses.

![Train and valid loss for NN2](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/df6a2c86-5781-4d01-85c4-aa761c41fb7e)



#### Comparing the performance
<br>
![image](https://github.com/Asin-30/Estimating-the-delivery-time-using-deep-learning-regression-models/assets/69243814/c27910c1-835e-4e81-ba23-e1427dc4900c)

We can see that first NN model i.e. *Neural Network 1* outperfromed all other models. It gave lease rmse (6%). This also shows that deep learning models have shown drastic improvements in terms of error reduction and enhance the model's accuracy.









