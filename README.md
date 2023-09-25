# Estimating-the-delivery-time-using-deep-learning-regression-models
This case study compares the performance of algorithms from classical machine learning like Bagging (Random Forest) and Deep Learning model.

## Objective
An online food delivery app has a number of delivery partners available for delivering the food, from various restaurants and wants to get an estimated delivery time that it can provide the customers on the basis of what they are ordering, from where and also the delivery partners.
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

